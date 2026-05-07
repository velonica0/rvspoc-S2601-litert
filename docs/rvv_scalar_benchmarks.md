# RVV Scalar Benchmark Results

- Date: 2026-05-06 20:20:06 CST
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
| zero_f32_64 | 64 | 4000 | 0.11 | 0.06 | 1.95 | 0.572 | 1.114 | 0 | 0 |
| zero_f32_1024 | 1024 | 4000 | 1.76 | 0.91 | 1.93 | 0.583 | 1.125 | 0 | 0 |
| zero_f32_4096 | 4096 | 4000 | 7.00 | 3.64 | 1.92 | 0.585 | 1.126 | 0 | 0 |

## IsZeroVector<int8>

- Affects: zero-skip checks for quantized helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero_i8_64 | 64 | 4000 | 0.03 | 0.02 | 2.10 | 1.915 | 4.019 | 0 | 0 |
| zero_i8_1024 | 1024 | 4000 | 0.47 | 0.22 | 2.14 | 2.173 | 4.645 | 0 | 0 |
| zero_i8_4096 | 4096 | 4000 | 1.89 | 0.88 | 2.13 | 2.171 | 4.631 | 0 | 0 |

## VectorVectorDotProduct<float>

- Affects: `fully_connected`, recurrent math, shared float tensor-utils call sites.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dot_64 | 64 | 4000 | 0.09 | 0.03 | 2.93 | 1.415 | 4.139 | 0.00000000 | 0.00000000 | 0.0000000000 |
| dot_256 | 256 | 4000 | 0.36 | 0.08 | 4.37 | 1.435 | 6.277 | 0.00000119 | 0.00000046 | 0.0000011921 |
| dot_1024 | 1024 | 4000 | 1.41 | 0.28 | 5.09 | 1.457 | 7.410 | 0.00000191 | 0.00000016 | 0.0000019073 |
| dot_4096 | 4096 | 4000 | 5.60 | 1.07 | 5.25 | 1.462 | 7.683 | 0.00002384 | 0.00000264 | 0.0000238419 |

## ReductionSumVector<float>

- Affects: `svdf` and other float reduction-style helper paths.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| svdf_small | 64 x 8 | 4000 | 1.00 | 0.77 | 1.30 | 0.514 | 0.666 | 0.00000036 | 0.00000050 | 0.0000000531 |
| svdf_mid | 128 x 16 | 4000 | 4.22 | 1.76 | 2.40 | 0.485 | 1.165 | 0.00000095 | 0.00000355 | 0.0000001383 |
| svdf_large | 256 x 64 | 4000 | 35.05 | 6.32 | 5.55 | 0.468 | 2.593 | 0.00000238 | 0.00010275 | 0.0000005155 |
| reduce_wide | 256 x 256 | 2441 | 141.54 | 17.54 | 8.07 | 0.463 | 3.736 | 0.00000954 | 0.00001902 | 0.0000022934 |

## ReductionSumVector<int8 -> int32>

- Affects: `conv`, `batch_matmul`, `kernel_utils`, cached row sum preparation.

| Case | Output x reduction | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| row_sum_small | 64 x 32 | 4000 | 1.09 | 1.07 | 1.02 | 1.883 | 1.913 | 0 | 0 |
| row_sum_mid | 128 x 128 | 4000 | 5.67 | 4.76 | 1.19 | 2.888 | 3.444 | 0 | 0 |
| row_sum_large | 256 x 256 | 2441 | 20.65 | 16.48 | 1.25 | 3.174 | 3.978 | 0 | 0 |
| row_sum_conv_like | 512 x 512 | 610 | 79.59 | 60.93 | 1.31 | 3.294 | 4.303 | 0 | 0 |

## MatrixScalarMultiplyAccumulate<int8>

- Affects: quantized recurrent helpers and row-wise reduction paths.

| Case | Rows x cols | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_rows_64 | 64 x 64 | 4000 | 4.11 | 1.59 | 2.59 | 0.997 | 2.582 | 0 | 0 |
| fc_rows_256 | 256 x 128 | 4000 | 30.46 | 9.79 | 3.11 | 1.076 | 3.346 | 0 | 0 |
| conv_rows_512 | 512 x 256 | 1220 | 122.47 | 33.51 | 3.65 | 1.070 | 3.911 | 0 | 0 |
| conv_rows_1024 | 1024 x 512 | 305 | 484.00 | 123.48 | 3.92 | 1.083 | 4.246 | 0 | 0 |

## MatrixBatchVectorMultiplyAccumulate<float>

- Affects: float `fully_connected` and recurrent kernels.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_fc | 64 x 64 x 4 | 4000 | 24.37 | 7.21 | 3.38 | 1.345 | 4.545 | 0.00000286 | 0.00010668 | 0.0000003323 |
| mid_fc | 128 x 128 x 4 | 2441 | 87.58 | 23.90 | 3.66 | 1.497 | 5.484 | 0.00000429 | 0.00002208 | 0.0000006108 |
| lstm_like | 640 x 2048 x 4 | 30 | 7210.50 | 2361.13 | 3.05 | 1.454 | 4.441 | 0.00011063 | 0.00308288 | 0.0000091973 |
| sqrnn_like | 1024 x 1024 x 8 | 20 | 11568.47 | 3748.31 | 3.09 | 1.450 | 4.476 | 0.00004292 | 0.00246700 | 0.0000045306 |

## MatrixBatchVectorMultiplyAccumulate<int8>

- Affects: quantized `fully_connected`, hybrid recurrent helpers, shared int8 GEMV-style call sites.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_fc_small | 64 x 64 x 4 | 4000 | 17.34 | 6.98 | 2.48 | 1.890 | 4.696 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_fc_mid | 128 x 128 x 4 | 2441 | 70.08 | 20.91 | 3.35 | 1.870 | 6.269 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_lstm_like | 640 x 1024 x 4 | 61 | 2787.36 | 593.87 | 4.69 | 1.881 | 8.828 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_conv_like | 1024 x 512 x 8 | 38 | 4497.86 | 1005.33 | 4.47 | 1.865 | 8.344 | 0.00000000 | 0.00000000 | 0.0000000000 |

## MatrixBatchVectorMultiplyAccumulate<int8, per-channel + input_offset>

- Affects: quantized `conv`, `batch_matmul`, and any path that uses cached row sums plus per-channel scale.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q_row_sum_small | 64 x 64 x 4 | 4000 | 26.14 | 8.63 | 3.03 | 1.253 | 3.798 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_mid | 128 x 128 x 4 | 2441 | 99.88 | 25.89 | 3.86 | 1.312 | 5.063 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_row_sum_conv | 256 x 576 x 8 | 135 | 1719.06 | 314.47 | 5.47 | 1.372 | 7.502 | 0.00000000 | 0.00000000 | 0.0000000000 |
| q_batch_matmul_like | 512 x 512 x 8 | 76 | 3047.26 | 566.06 | 5.38 | 1.376 | 7.410 | 0.00000000 | 0.00000000 | 0.0000000000 |

## MatrixBatchVectorMultiplyAccumulate<int8 -> int16>

- Affects: quantized recurrent gate accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q15_small | 128 x 128 x 4 | 2441 | 76.61 | 21.46 | 3.57 | 1.711 | 6.108 | 0 | 0 |
| gate_q15_mid | 256 x 256 x 4 | 610 | 300.37 | 71.04 | 4.23 | 1.745 | 7.380 | 0 | 0 |
| gate_q15_large | 512 x 512 x 8 | 76 | 2293.13 | 507.21 | 4.52 | 1.829 | 8.269 | 0 | 0 |

## MatrixBatchVectorMultiplyAccumulate<int8 -> int8>

- Affects: quantized projection and low-precision recurrent accumulate helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_q8_small | 128 x 128 x 4 | 2441 | 77.39 | 21.67 | 3.57 | 1.694 | 6.048 | 0 | 0 |
| gate_q8_mid | 256 x 256 x 4 | 610 | 304.94 | 71.33 | 4.28 | 1.719 | 7.350 | 0 | 0 |
| gate_q8_large | 512 x 512 x 8 | 76 | 2242.23 | 510.53 | 4.39 | 1.871 | 8.216 | 0 | 0 |

## SparseMatrixBatchVectorMultiplyAccumulate1x4<float>

- Affects: sparse float `fully_connected` and sparse recurrent helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_1x4_small | 64 x 256 x 4 | 4000 | 22.48 | 11.99 | 1.87 | 1.455 | 2.728 | 0.00000191 | 0.00003969 | 0.0000003132 |
| sparse_1x4_mid | 128 x 512 x 4 | 2392 | 99.52 | 50.97 | 1.95 | 1.344 | 2.624 | 0.00000477 | 0.00003892 | 0.0000006231 |
| sparse_1x4_large | 256 x 1024 x 8 | 303 | 760.25 | 374.13 | 2.03 | 1.386 | 2.817 | 0.00000954 | 0.00044747 | 0.0000012181 |

## SparseMatrixBatchVectorMultiplyAccumulate<float ledger>

- Affects: sparse float matvec helper paths with ledger format.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_ledger_small | 64 x 256 x 4 | 4000 | 23.38 | 11.20 | 2.09 | 1.287 | 2.685 | 0.00000286 | 0.00000875 | 0.0000003210 |
| sparse_ledger_mid | 128 x 512 x 4 | 2347 | 103.64 | 42.45 | 2.44 | 1.315 | 3.212 | 0.00000429 | 0.00013315 | 0.0000006498 |
| sparse_ledger_large | 256 x 1024 x 8 | 303 | 809.27 | 293.58 | 2.76 | 1.302 | 3.588 | 0.00001240 | 0.00009487 | 0.0000011948 |

## SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>

- Affects: sparse quantized matvec helper paths that dequantize to float.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_qfloat_small | 64 x 256 x 4 | 4000 | 15.63 | 10.35 | 1.51 | 2.096 | 3.165 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_mid | 128 x 512 x 4 | 2485 | 58.27 | 32.57 | 1.79 | 2.210 | 3.954 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sparse_qfloat_large | 256 x 1024 x 8 | 303 | 476.62 | 230.65 | 2.07 | 2.210 | 4.567 | 0.00000000 | 0.00000000 | 0.0000000000 |

## SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>

- Affects: sparse quantized fully-connected and recurrent output helper paths.

| Case | Rows x cols x batch | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse_q8_small | 64 x 256 x 4 | 4000 | 32.31 | 15.33 | 2.11 | 1.006 | 2.120 | 0 | 0 |
| sparse_q8_mid | 128 x 512 x 4 | 2431 | 122.84 | 46.89 | 2.62 | 1.071 | 2.806 | 0 | 0 |
| sparse_q8_large | 256 x 1024 x 8 | 301 | 952.80 | 288.19 | 3.31 | 1.115 | 3.686 | 0 | 0 |

## SymmetricQuantizeFloats

- Affects: hybrid `fully_connected`, `batch_matmul`, weight and activation pre-quant helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Min diff | Max diff | Scale diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sym_quant_small | 64 | 4000 | 0.47 | 0.16 | 2.94 | 0.412 | 1.212 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_mid | 1024 | 4000 | 7.21 | 2.67 | 2.70 | 0.426 | 1.151 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| sym_quant_large | 4096 | 4000 | 32.14 | 14.88 | 2.16 | 0.382 | 0.826 | 0 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |

## AsymmetricQuantizeFloats

- Affects: hybrid `conv`, `depthwise_conv`, `transpose_conv`, and int8 input staging paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches | Scale diff | Offset diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asym_quant_small | 64 | 4000 | 0.56 | 0.24 | 2.38 | 0.341 | 0.812 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_mid | 1024 | 4000 | 8.65 | 3.27 | 2.65 | 0.355 | 0.940 | 0 | 0 | 0.00000000 | 0 |
| asym_quant_large | 4096 | 4000 | 37.68 | 16.23 | 2.32 | 0.326 | 0.757 | 0 | 0 | 0.00000000 | 0 |

## ApplyLayerNorm<int16>

- Affects: quantized recurrent gate normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ln_small | 4 x 64 | 4000 | 6.00 | 2.39 | 2.51 | 0.342 | 0.858 | 0 | 0 |
| ln_mid | 4 x 256 | 4000 | 24.04 | 8.65 | 2.78 | 0.341 | 0.947 | 0 | 0 |
| ln_gate_like | 8 x 1024 | 2441 | 193.53 | 67.48 | 2.87 | 0.339 | 0.971 | 0 | 0 |

## ApplySigmoid<int16>

- Affects: quantized recurrent gate activation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigmoid_small | 4 x 64 | 4000 | 22.20 | 8.63 | 2.57 | 0.012 | 0.030 | 0 | 0 |
| sigmoid_mid | 8 x 256 | 4000 | 192.81 | 68.98 | 2.80 | 0.011 | 0.030 | 0 | 0 |
| sigmoid_large | 4 x 1024 | 4000 | 387.00 | 137.74 | 2.81 | 0.011 | 0.030 | 0 | 0 |

## ApplyTanh<int16>

- Affects: quantized recurrent state activation helper paths.

| Case | Bits x batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tanh_q0_small | 0 x 4 x 64 | 4000 | 21.20 | 8.22 | 2.58 | 0.012 | 0.031 | 0 | 0 |
| tanh_q3_mid | 3 x 8 x 256 | 4000 | 200.22 | 71.89 | 2.79 | 0.010 | 0.028 | 0 | 0 |
| tanh_q4_large | 4 x 4 x 1024 | 4000 | 405.93 | 150.42 | 2.70 | 0.010 | 0.027 | 0 | 0 |

## CwiseMul<int16 -> int16>

- Affects: quantized recurrent elementwise gate math.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q15_small | 4 x 64 | 4000 | 0.57 | 0.44 | 1.30 | 0.450 | 0.585 | 0 | 0 |
| mul_q15_mid | 8 x 256 | 4000 | 4.34 | 3.46 | 1.25 | 0.472 | 0.593 | 0 | 0 |
| mul_q15_large | 4 x 1024 | 4000 | 8.63 | 6.92 | 1.25 | 0.475 | 0.592 | 0 | 0 |

## CwiseMul<int16 -> int8>

- Affects: quantized projection and low-precision elementwise paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_q8_small | 4 x 64 | 4000 | 2.52 | 0.77 | 3.28 | 0.101 | 0.333 | 0 | 0 |
| mul_q8_mid | 8 x 256 | 4000 | 23.82 | 6.03 | 3.95 | 0.086 | 0.340 | 0 | 0 |
| mul_q8_large | 4 x 1024 | 4000 | 49.47 | 12.07 | 4.10 | 0.083 | 0.339 | 0 | 0 |

## CwiseAdd<int16>

- Affects: recurrent residual/additive helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_small | 4 x 64 | 4000 | 0.73 | 0.33 | 2.20 | 0.350 | 0.769 | 0 | 0 |
| add_mid | 8 x 256 | 4000 | 6.37 | 2.62 | 2.43 | 0.321 | 0.781 | 0 | 0 |
| add_large | 4 x 1024 | 4000 | 14.72 | 5.25 | 2.80 | 0.278 | 0.780 | 0 | 0 |

## CwiseClipping<float>

- Affects: activation clamp paths in float recurrent/helper code.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_f32_64 | 64 | 4000 | 0.07 | 0.05 | 1.49 | 0.879 | 1.314 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_1024 | 1024 | 4000 | 1.17 | 0.76 | 1.54 | 0.877 | 1.354 | 0.00000000 | 0.00000000 | 0.0000000000 |
| clip_f32_4096 | 4096 | 4000 | 4.67 | 3.04 | 1.54 | 0.877 | 1.347 | 0.00000000 | 0.00000000 | 0.0000000000 |

## CwiseClipping<int16>

- Affects: quantized activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i16_64 | 64 | 4000 | 0.02 | 0.02 | 1.00 | 3.193 | 3.197 | 0 | 0 |
| clip_i16_1024 | 1024 | 4000 | 0.32 | 0.32 | 1.00 | 3.186 | 3.200 | 0 | 0 |
| clip_i16_4096 | 4096 | 4000 | 1.29 | 1.29 | 1.00 | 3.174 | 3.168 | 0 | 0 |

## CwiseClipping<int8>

- Affects: low-precision activation clamp helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clip_i8_64 | 64 | 4000 | 0.01 | 0.02 | 0.96 | 4.297 | 4.104 | 0 | 0 |
| clip_i8_1024 | 1024 | 4000 | 0.16 | 0.16 | 1.00 | 6.399 | 6.400 | 0 | 0 |
| clip_i8_4096 | 4096 | 4000 | 0.64 | 0.64 | 1.00 | 6.378 | 6.400 | 0 | 0 |

## VectorBatchVectorCwiseProductAccumulate<int16>

- Affects: quantized recurrent gate/state accumulation helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vbv_cwise_small | 4 x 64 | 4000 | 2.35 | 0.81 | 2.92 | 0.218 | 0.635 | 0 | 0 |
| vbv_cwise_mid | 8 x 256 | 4000 | 23.23 | 6.31 | 3.68 | 0.176 | 0.649 | 0 | 0 |
| vbv_cwise_large | 4 x 1024 | 4000 | 47.55 | 12.65 | 3.76 | 0.172 | 0.648 | 0 | 0 |

## Sub1Vector<float>

- Affects: float recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_f32_64 | 64 | 4000 | 0.06 | 0.04 | 1.72 | 1.020 | 1.759 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_1024 | 1024 | 4000 | 0.94 | 0.58 | 1.61 | 1.087 | 1.754 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub1_f32_4096 | 4096 | 4000 | 3.75 | 2.34 | 1.60 | 1.091 | 1.747 | 0.00000000 | 0.00000000 | 0.0000000000 |

## Sub1Vector<int16>

- Affects: quantized recurrent helper post-processing paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub1_i16_64 | 64 | 4000 | 0.06 | 0.02 | 3.37 | 1.043 | 3.515 | 0 | 0 |
| sub1_i16_1024 | 1024 | 4000 | 0.94 | 0.29 | 3.24 | 1.087 | 3.520 | 0 | 0 |
| sub1_i16_4096 | 4096 | 4000 | 3.75 | 1.18 | 3.18 | 1.092 | 3.473 | 0 | 0 |

## VectorScalarMultiply<int8 -> float>

- Affects: hybrid dequant-style helper paths.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scale_i8_64 | 64 | 4000 | 0.09 | 0.03 | 3.55 | 1.415 | 5.022 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_1024 | 1024 | 4000 | 1.41 | 0.33 | 4.25 | 1.452 | 6.169 | 0.00000000 | 0.00000000 | 0.0000000000 |
| scale_i8_4096 | 4096 | 4000 | 5.63 | 1.28 | 4.39 | 1.454 | 6.384 | 0.00000000 | 0.00000000 | 0.0000000000 |

## MeanStddevNormalization<float>

- Affects: float recurrent normalization helper paths.

| Case | Batch x input | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| norm_1x64 | 1 x 64 | 4000 | 0.27 | 0.21 | 1.27 | 0.943 | 1.199 | 0.00000036 | 0.00000024 | 0.0000001536 |
| norm_4x256 | 4 x 256 | 4000 | 4.00 | 2.88 | 1.39 | 1.025 | 1.423 | 0.00000036 | 0.00000812 | 0.0000000711 |
| norm_8x1024 | 8 x 1024 | 4000 | 30.29 | 21.98 | 1.38 | 1.082 | 1.491 | 0.00000048 | 0.00003007 | 0.0000000807 |

