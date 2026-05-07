# RVV Scalar Benchmark Results

- Date: 2026-05-07 11:07:06 CST
- Host: Linux k3 6.18.3-generic #1.0.0~rc4.4 SMP PREEMPT_DYNAMIC Wed Apr 29 10:51:17 CST 2026 riscv64 GNU/Linux
- Compiler: g++ (Bianbu 15.2.0-16ubuntu1bb2) 15.2.0
- Build dir: `/home/openkylin/github/rvspoc-S2601-litert/build-riscv-bench`

# RVV vs Scalar: Shared tensor_utils coverage

- Runtime VLEN bits: 256
- Scalar kernels: `Portable*` compiled with `-fno-tree-vectorize -fno-tree-slp-vectorize`
- RVV kernels: `Rvv*` compiled with `-march=rv64gcv_zvl128b -mabi=lp64d`
- Covered helper families: zero-check, dot/reduction, dense matvec, sparse matvec, hybrid quantization, recurrent accumulate, layer-norm, cwise math, clipping, sub1, normalization, dequant-scale.

## IsZeroVector<float>

- Affects: zero-skip precheck paths in hybrid and recurrent helpers.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_f32_64 | 64 | 4000 | 0.11 | 0.06 | 1.96 | 0.569 | 1.114 | 0 | 0 |
| zero_f32_1024 | 1024 | 4000 | 1.77 | 0.91 | 1.95 | 0.578 | 1.126 | 0 | 0 |
| zero_f32_4096 | 4096 | 4000 | 7.00 | 3.62 | 1.93 | 0.585 | 1.130 | 0 | 0 |

## IsZeroVector<int8>

- Affects: zero-skip checks for quantized helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_i8_64 | 64 | 4000 | 0.03 | 0.02 | 2.00 | 2.007 | 4.019 | 0 | 0 |
| zero_i8_1024 | 1024 | 4000 | 0.47 | 0.22 | 2.14 | 2.167 | 4.645 | 0 | 0 |
| zero_i8_4096 | 4096 | 4000 | 1.89 | 0.88 | 2.13 | 2.171 | 4.633 | 0 | 0 |

## VectorVectorDotProduct<float>

- Affects: `fully_connected`, recurrent math, shared float tensor-utils call sites.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dot_64 | 64 | 4000 | 0.09 | 0.03 | 2.72 | 1.422 | 3.866 | 0.00000000 | 0.00000000 | 0.0000000000 |
| dot_256 | 256 | 4000 | 0.36 | 0.08 | 4.46 | 1.435 | 6.399 | 0.00000119 | 0.00000046 | 0.0000011921 |
| dot_1024 | 1024 | 4000 | 1.41 | 0.28 | 5.01 | 1.456 | 7.292 | 0.00000191 | 0.00000016 | 0.0000019073 |
| dot_4096 | 4096 | 4000 | 5.61 | 1.07 | 5.26 | 1.461 | 7.681 | 0.00002384 | 0.00000264 | 0.0000238419 |

## BatchVectorBatchVectorDotProduct<int16>

- Affects: `svdf` and batched recurrent dot-product helper paths.

| Case | Batch x size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_dot_q15_small | 4 x 64 | 4000 | 0.29 | 0.09 | 3.11 | 1.788 | 5.562 | 0 | 0 |
| svdf_dot_q15_mid | 8 x 256 | 4000 | 2.12 | 0.49 | 4.32 | 1.936 | 8.359 | 0 | 0 |
| svdf_dot_q15_large | 16 x 1024 | 4000 | 16.73 | 3.67 | 4.56 | 1.958 | 8.940 | 0 | 0 |

## ReductionSumVector<float>

- Affects: `svdf` and other float reduction-style helper paths.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_small | 64 x 8 | 4000 | 1.00 | 0.77 | 1.30 | 0.513 | 0.666 | 0.00000048 | 0.00000080 | 0.0000000559 |
| svdf_mid | 128 x 16 | 4000 | 4.23 | 1.76 | 2.40 | 0.484 | 1.164 | 0.00000095 | 0.00001488 | 0.0000001378 |
| svdf_large | 256 x 64 | 4000 | 35.07 | 6.32 | 5.55 | 0.467 | 2.592 | 0.00000572 | 0.00001453 | 0.0000005609 |
| reduce_wide | 256 x 256 | 2441 | 141.43 | 17.56 | 8.06 | 0.463 | 3.733 | 0.00001144 | 0.00016069 | 0.0000021656 |

## ReductionSumVector<int8 -> int32>

- Affects: `conv`, `batch_matmul`, `kernel_utils`, cached row sum preparation.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row_sum_small | 64 x 32 | 4000 | 1.09 | 1.07 | 1.02 | 1.879 | 1.921 | 0 | 0 |
| row_sum_mid | 128 x 128 | 4000 | 5.67 | 4.73 | 1.20 | 2.892 | 3.465 | 0 | 0 |
| row_sum_large | 256 x 256 | 2441 | 20.65 | 16.60 | 1.24 | 3.174 | 3.948 | 0 | 0 |
| row_sum_conv_like | 512 x 512 | 610 | 79.57 | 61.19 | 1.30 | 3.294 | 4.284 | 0 | 0 |

## ReductionSumVector<int32>

- Affects: scalar accumulation helper paths such as reference SVDF and utility reductions.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reduce_i32_small | 64 x 32 | 4000 | 0.91 | 0.77 | 1.19 | 2.244 | 2.667 | 0 | 0 |
| reduce_i32_mid | 128 x 128 | 4000 | 4.64 | 3.01 | 1.54 | 3.532 | 5.447 | 0 | 0 |
| reduce_i32_large | 256 x 256 | 2441 | 16.72 | 10.59 | 1.58 | 3.919 | 6.189 | 0 | 0 |

## MatrixScalarMultiplyAccumulate<int8>

- Affects: quantized recurrent helpers and row-wise reduction paths.

| Case | Rows x cols | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_rows_64 | 64 x 64 | 4000 | 4.11 | 1.58 | 2.60 | 0.997 | 2.588 | 0 | 0 |
| fc_rows_256 | 256 x 128 | 4000 | 30.48 | 9.79 | 3.11 | 1.075 | 3.346 | 0 | 0 |
| conv_rows_512 | 512 x 256 | 1220 | 122.41 | 33.53 | 3.65 | 1.071 | 3.909 | 0 | 0 |
| conv_rows_1024 | 1024 x 512 | 305 | 484.22 | 122.93 | 3.94 | 1.083 | 4.265 | 0 | 0 |

## MatrixBatchVectorMultiplyAccumulate<float>

- Affects: float `fully_connected` and recurrent kernels.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_fc | 64 x 64 x 4 | 4000 | 24.25 | 7.26 | 3.34 | 1.351 | 4.514 | 0.00000286 | 0.00001540 | 0.0000003163 |
| mid_fc | 128 x 128 x 4 | 2441 | 87.59 | 23.99 | 3.65 | 1.497 | 5.463 | 0.00000525 | 0.00006299 | 0.0000006228 |
| lstm_like | 640 x 2048 x 4 | 30 | 7209.80 | 2371.16 | 3.04 | 1.454 | 4.422 | 0.00008774 | 0.00183174 | 0.0000094324 |
| sqrnn_like | 1024 x 1024 x 8 | 20 | 11564.11 | 3846.86 | 3.01 | 1.451 | 4.361 | 0.00005531 | 0.00156548 | 0.0000045446 |

## MatrixBatchVectorMultiplyAccumulate<int8>

- Affects: quantized `fully_connected`, hybrid recurrent helpers, shared int8 GEMV-style call sites.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_fc_small | 64 x 64 x 4 | 4000 | 16.51 | 6.97 | 2.37 | 1.985 | 4.698 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_fc_mid | 128 x 128 x 4 | 2441 | 70.46 | 20.90 | 3.37 | 1.860 | 6.272 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_lstm_like | 640 x 1024 x 4 | 61 | 2788.09 | 594.63 | 4.69 | 1.880 | 8.817 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_conv_like | 1024 x 512 x 8 | 38 | 4469.24 | 1004.73 | 4.45 | 1.877 | 8.349 | 0.00000000 | 0.00000000 | 0.0000000000 |

## MatrixBatchVectorMultiplyAccumulate<int8, per-channel + input_offset>

- Affects: quantized `conv`, `batch_matmul`, and any path that uses cached row sums plus per-channel scale.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_row_sum_small | 64 x 64 x 4 | 4000 | 26.03 | 8.62 | 3.02 | 1.259 | 3.801 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_mid | 128 x 128 x 4 | 2441 | 100.02 | 25.91 | 3.86 | 1.310 | 5.058 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_conv | 256 x 576 x 8 | 135 | 1720.83 | 314.90 | 5.46 | 1.371 | 7.492 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_batch_matmul_like | 512 x 512 x 8 | 76 | 3064.73 | 565.96 | 5.42 | 1.369 | 7.411 | 0.00000000 | 0.00000000 | 0.0000000000 |

## MatrixBatchVectorMultiplyAccumulate<int8 -> int16>

- Affects: quantized recurrent gate accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q15_small | 128 x 128 x 4 | 2441 | 76.43 | 21.49 | 3.56 | 1.715 | 6.099 | 0 | 0 |
| gate_q15_mid | 256 x 256 x 4 | 610 | 287.55 | 71.18 | 4.04 | 1.823 | 7.366 | 0 | 0 |
| gate_q15_large | 512 x 512 x 8 | 76 | 2298.91 | 508.06 | 4.52 | 1.824 | 8.256 | 0 | 0 |

## MatrixBatchVectorMultiplyAccumulate<int8 -> int8>

- Affects: quantized projection and low-precision recurrent accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q8_small | 128 x 128 x 4 | 2441 | 77.69 | 21.63 | 3.59 | 1.687 | 6.061 | 0 | 0 |
| gate_q8_mid | 256 x 256 x 4 | 610 | 292.18 | 71.41 | 4.09 | 1.794 | 7.342 | 0 | 0 |
| gate_q8_large | 512 x 512 x 8 | 76 | 2252.55 | 508.76 | 4.43 | 1.862 | 8.244 | 0 | 0 |

## MatrixBatchVectorMultiply<int8 -> int8>

- Affects: quantized LSTM gate matmul paths before saturating-add and activation.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_noacc_q8_small | 128 x 128 x 4 | 2441 | 82.95 | 30.55 | 2.72 | 1.580 | 4.291 | 0 | 0 |
| gate_noacc_q8_mid | 256 x 256 x 4 | 610 | 300.19 | 100.69 | 2.98 | 1.747 | 5.207 | 0 | 0 |
| gate_noacc_q8_large | 512 x 512 x 8 | 76 | 2220.38 | 701.12 | 3.17 | 1.889 | 5.982 | 0 | 0 |

## MatrixBatchVectorMultiply<int16 x int8 -> int8>

- Affects: quantized projection/output matmul helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| proj_q8_small | 128 x 128 x 4 | 2441 | 422.41 | 32.33 | 13.07 | 0.310 | 4.054 | 0 | 0 |
| proj_q8_mid | 256 x 256 x 4 | 610 | 1662.87 | 117.48 | 14.15 | 0.315 | 4.463 | 0 | 0 |
| proj_q8_large | 512 x 512 x 8 | 76 | 13212.60 | 894.91 | 14.76 | 0.317 | 4.687 | 0 | 0 |

## SparseMatrixBatchVectorMultiplyAccumulate1x4<float>

- Affects: sparse float `fully_connected` and sparse recurrent helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_1x4_small | 64 x 256 x 4 | 4000 | 21.91 | 11.85 | 1.85 | 1.481 | 2.739 | 0.00000143 | 0.00012937 | 0.0000003011 |
| sparse_1x4_mid | 128 x 512 x 4 | 2448 | 98.16 | 50.07 | 1.96 | 1.331 | 2.610 | 0.00000477 | 0.00038041 | 0.0000006075 |
| sparse_1x4_large | 256 x 1024 x 8 | 304 | 754.86 | 373.80 | 2.02 | 1.391 | 2.810 | 0.00000954 | 0.00038029 | 0.0000011599 |

## SparseMatrixBatchVectorMultiplyAccumulate<float ledger>

- Affects: sparse float matvec helper paths with ledger format.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_ledger_small | 64 x 256 x 4 | 4000 | 24.99 | 11.69 | 2.14 | 1.291 | 2.759 | 0.00000238 | 0.00000709 | 0.0000003208 |
| sparse_ledger_mid | 128 x 512 x 4 | 2362 | 103.75 | 41.62 | 2.49 | 1.305 | 3.254 | 0.00000572 | 0.00038695 | 0.0000006152 |
| sparse_ledger_large | 256 x 1024 x 8 | 303 | 810.13 | 294.68 | 2.75 | 1.303 | 3.584 | 0.00001240 | 0.00644636 | 0.0000011960 |

## SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>

- Affects: sparse quantized matvec helper paths that dequantize to float.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_qfloat_small | 64 x 256 x 4 | 4000 | 15.64 | 10.39 | 1.50 | 2.071 | 3.116 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_mid | 128 x 512 x 4 | 2465 | 58.67 | 32.94 | 1.78 | 2.212 | 3.940 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_large | 256 x 1024 x 8 | 299 | 483.05 | 234.40 | 2.06 | 2.212 | 4.559 | 0.00000000 | 0.00000000 | 0.0000000000 |

## SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>

- Affects: sparse quantized fully-connected and recurrent output helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_q8_small | 64 x 256 x 4 | 4000 | 33.86 | 15.80 | 2.14 | 1.009 | 2.162 | 0 | 0 |
| sparse_q8_mid | 128 x 512 x 4 | 2394 | 124.60 | 46.91 | 2.66 | 1.072 | 2.849 | 0 | 0 |
| sparse_q8_large | 256 x 1024 x 8 | 302 | 944.10 | 286.51 | 3.30 | 1.121 | 3.693 | 0 | 0 |

## SymmetricQuantizeFloats

- Affects: hybrid `fully_connected`, `batch_matmul`, weight and activation pre-quant helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Min diff | Max diff | Scale diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sym_quant_small | 64 | 4000 | 0.46 | 0.15 | 2.98 | 0.419 | 1.246 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_mid | 1024 | 4000 | 7.37 | 2.65 | 2.78 | 0.417 | 1.158 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_large | 4096 | 4000 | 32.41 | 15.08 | 2.15 | 0.379 | 0.815 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

## AsymmetricQuantizeFloats

- Affects: hybrid `conv`, `depthwise_conv`, `transpose_conv`, and int8 input staging paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Scale diff | Offset diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asym_quant_small | 64 | 4000 | 0.58 | 0.25 | 2.35 | 0.330 | 0.776 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_mid | 1024 | 4000 | 8.52 | 3.31 | 2.57 | 0.361 | 0.927 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_large | 4096 | 4000 | 37.29 | 16.35 | 2.28 | 0.329 | 0.751 | 0 | 0 | 0.00000000 | 0 |

## ApplyLayerNorm<int16>

- Affects: quantized recurrent gate normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ln_small | 4 x 64 | 4000 | 5.90 | 2.39 | 2.47 | 0.347 | 0.857 | 0 | 0 |
| ln_mid | 4 x 256 | 4000 | 24.04 | 8.66 | 2.78 | 0.341 | 0.946 | 0 | 0 |
| ln_gate_like | 8 x 1024 | 2441 | 193.18 | 67.50 | 2.86 | 0.339 | 0.971 | 0 | 0 |

## ApplySigmoid<int16>

- Affects: quantized recurrent gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoid_small | 4 x 64 | 4000 | 22.31 | 8.63 | 2.59 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_mid | 8 x 256 | 4000 | 192.44 | 68.89 | 2.79 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_large | 4 x 1024 | 4000 | 386.38 | 137.82 | 2.80 | 0.011 | 0.030 | 0 | 0 |

## ApplyTanh<int16>

- Affects: quantized recurrent state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanh_q0_small | 0 x 4 x 64 | 4000 | 21.26 | 8.26 | 2.57 | 0.012 | 0.031 | 0 | 0 |
| tanh_q3_mid | 3 x 8 x 256 | 4000 | 200.80 | 71.81 | 2.80 | 0.010 | 0.029 | 0 | 0 |
| tanh_q4_large | 4 x 4 x 1024 | 4000 | 405.34 | 150.41 | 2.69 | 0.010 | 0.027 | 0 | 0 |

## ApplyLayerNormFloat<int16>

- Affects: float-reference layer-norm helper paths used by quantized LSTM eval.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lnf_small | 4 x 64 | 4000 | 3.69 | 1.47 | 2.51 | 0.554 | 1.390 | 0 | 0 |
| lnf_mid | 4 x 256 | 4000 | 13.92 | 5.30 | 2.63 | 0.588 | 1.544 | 0 | 0 |
| lnf_gate_like | 8 x 1024 | 2441 | 109.41 | 40.92 | 2.67 | 0.599 | 1.601 | 0 | 0 |

## ApplySigmoidFloat<int16>

- Affects: float-reference gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoidf_small | 4 x 64 | 4000 | 9.39 | 8.63 | 1.09 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 2.0703125000 |
| sigmoidf_mid | 8 x 256 | 4000 | 74.76 | 68.91 | 1.08 | 0.027 | 0.030 | 5.00000000 | 0.21052632 | 1.9697265625 |
| sigmoidf_large | 4 x 1024 | 4000 | 149.36 | 137.83 | 1.08 | 0.027 | 0.030 | 6.00000000 | 0.21052632 | 1.9594726562 |

## ApplyTanhFloat<int16>

- Affects: float-reference state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanhf_qm12_small | -12 x 4 x 64 | 4000 | 24.77 | 8.89 | 2.79 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm12_mid | -12 x 8 x 256 | 4000 | 197.84 | 70.19 | 2.82 | 0.010 | 0.029 | 0 | 0 |
| tanhf_qm15_large | -15 x 4 x 1024 | 4000 | 395.65 | 140.19 | 2.82 | 0.010 | 0.029 | 0 | 0 |

## CwiseMul<int16 -> int16>

- Affects: quantized recurrent elementwise gate math.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q15_small | 4 x 64 | 4000 | 0.57 | 0.44 | 1.31 | 0.448 | 0.585 | 0 | 0 |
| mul_q15_mid | 8 x 256 | 4000 | 4.33 | 3.48 | 1.24 | 0.473 | 0.589 | 0 | 0 |
| mul_q15_large | 4 x 1024 | 4000 | 8.71 | 6.93 | 1.26 | 0.470 | 0.591 | 0 | 0 |

## CwiseMul<int16 -> int8>

- Affects: quantized projection and low-precision elementwise paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q8_small | 4 x 64 | 4000 | 2.49 | 0.77 | 3.23 | 0.103 | 0.332 | 0 | 0 |
| mul_q8_mid | 8 x 256 | 4000 | 23.74 | 6.02 | 3.94 | 0.086 | 0.340 | 0 | 0 |
| mul_q8_large | 4 x 1024 | 4000 | 49.37 | 12.05 | 4.10 | 0.083 | 0.340 | 0 | 0 |

## CwiseAdd<int16>

- Affects: recurrent residual/additive helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_small | 4 x 64 | 4000 | 0.71 | 0.33 | 2.13 | 0.362 | 0.770 | 0 | 0 |
| add_mid | 8 x 256 | 4000 | 6.38 | 2.62 | 2.44 | 0.321 | 0.782 | 0 | 0 |
| add_large | 4 x 1024 | 4000 | 15.14 | 5.28 | 2.86 | 0.271 | 0.775 | 0 | 0 |

## TwoGateSaturatingAdd<int8 -> int16>

- Affects: quantized LSTM gate merge helper paths.

| Case | Batch x cell | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| two_gate_small | 4 x 64 | 4000 | 4.21 | 1.30 | 3.24 | 0.243 | 0.786 | 0 | 0 |
| two_gate_mid | 8 x 256 | 4000 | 42.38 | 10.25 | 4.13 | 0.193 | 0.799 | 0 | 0 |
| two_gate_large | 4 x 1024 | 4000 | 84.21 | 20.45 | 4.12 | 0.195 | 0.801 | 0 | 0 |

## CwiseClipping<float>

- Affects: activation clamp paths in float recurrent/helper code.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_f32_64 | 64 | 4000 | 0.07 | 0.05 | 1.54 | 0.879 | 1.353 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_1024 | 1024 | 4000 | 1.17 | 0.76 | 1.54 | 0.879 | 1.352 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_4096 | 4096 | 4000 | 4.68 | 3.04 | 1.54 | 0.876 | 1.348 | 0.00000000 | 0.00000000 | 0.0000000000 |

## CwiseClipping<int16>

- Affects: quantized activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i16_64 | 64 | 4000 | 0.02 | 0.02 | 1.00 | 3.195 | 3.197 | 0 | 0 |
| clip_i16_1024 | 1024 | 4000 | 0.32 | 0.32 | 0.99 | 3.200 | 3.183 | 0 | 0 |
| clip_i16_4096 | 4096 | 4000 | 1.29 | 1.30 | 1.00 | 3.173 | 3.161 | 0 | 0 |

## CwiseClipping<int8>

- Affects: low-precision activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i8_64 | 64 | 4000 | 0.01 | 0.02 | 0.93 | 4.462 | 4.163 | 0 | 0 |
| clip_i8_1024 | 1024 | 4000 | 0.16 | 0.16 | 0.99 | 6.399 | 6.345 | 0 | 0 |
| clip_i8_4096 | 4096 | 4000 | 0.64 | 0.64 | 1.00 | 6.400 | 6.371 | 0 | 0 |

## VectorBatchVectorCwiseProductAccumulate<int16>

- Affects: quantized recurrent gate/state accumulation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vbv_cwise_small | 4 x 64 | 4000 | 2.29 | 0.81 | 2.84 | 0.223 | 0.635 | 0 | 0 |
| vbv_cwise_mid | 8 x 256 | 4000 | 23.19 | 6.31 | 3.68 | 0.177 | 0.649 | 0 | 0 |
| vbv_cwise_large | 4 x 1024 | 4000 | 47.29 | 12.60 | 3.75 | 0.173 | 0.650 | 0 | 0 |

## Sub1Vector<float>

- Affects: float recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_f32_64 | 64 | 4000 | 0.06 | 0.04 | 1.74 | 1.013 | 1.759 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_1024 | 1024 | 4000 | 0.95 | 0.58 | 1.63 | 1.082 | 1.760 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_4096 | 4096 | 4000 | 3.74 | 2.34 | 1.60 | 1.095 | 1.748 | 0.00000000 | 0.00000000 | 0.0000000000 |

## Sub1Vector<int16>

- Affects: quantized recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_i16_64 | 64 | 4000 | 0.06 | 0.02 | 3.37 | 1.043 | 3.515 | 0 | 0 |
| sub1_i16_1024 | 1024 | 4000 | 0.94 | 0.29 | 3.21 | 1.089 | 3.502 | 0 | 0 |
| sub1_i16_4096 | 4096 | 4000 | 3.75 | 1.18 | 3.18 | 1.092 | 3.473 | 0 | 0 |

## VectorScalarMultiply<int8 -> float>

- Affects: hybrid dequant-style helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale_i8_64 | 64 | 4000 | 0.09 | 0.02 | 4.30 | 1.422 | 6.111 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_1024 | 1024 | 4000 | 1.41 | 0.32 | 4.41 | 1.450 | 6.391 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_4096 | 4096 | 4000 | 5.61 | 1.28 | 4.38 | 1.460 | 6.392 | 0.00000000 | 0.00000000 | 0.0000000000 |

## MeanStddevNormalization<float>

- Affects: float recurrent normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| norm_1x64 | 1 x 64 | 4000 | 0.27 | 0.21 | 1.29 | 0.938 | 1.211 | 0.00000024 | 0.00000023 | 0.0000000843 |
| norm_4x256 | 4 x 256 | 4000 | 3.95 | 2.88 | 1.37 | 1.038 | 1.425 | 0.00000048 | 0.00002272 | 0.0000001109 |
| norm_8x1024 | 8 x 1024 | 4000 | 30.29 | 21.90 | 1.38 | 1.082 | 1.496 | 0.00000083 | 0.00009705 | 0.0000001304 |

