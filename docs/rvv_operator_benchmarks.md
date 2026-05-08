# RVV Operator Benchmark Results

- Date: 2026-05-08 14:56:48 CST
- Host: Linux k3 6.18.3-generic #1.0.0~rc4.4 SMP PREEMPT_DYNAMIC Wed Apr 29 10:51:17 CST 2026 riscv64 GNU/Linux
- Compiler: g++ (Bianbu 15.2.0-16ubuntu1bb2) 15.2.0
- Build dir: `/home/openkylin/github/rvspoc-S2601-litert/build-riscv-op-bench`

# RVV vs Scalar: Operator-level RVV coverage

- Scalar kernels: same wrapper compiled with `-fno-tree-vectorize -fno-tree-slp-vectorize`
- RVV kernels: same wrapper compiled with `-march=rv64gcv_zvl128b -mabi=lp64d`
- Organization: benchmark sections are split by the source file that owns the RVV path.
- Scope: includes the earlier recent-five-commit operator set plus follow-up P1 gap closures for `reduce.h` and `resize_bilinear.h`.

## `tflite/kernels/internal/optimized/optimized_ops.h`

- Summary: Float arithmetic fast paths, quantize/dequant-style helpers, uint8 pooling, quantized HardSwish, and argmin/argmax cases that gained RVV coverage in the latest operator-level commits.

### Add<float>

- Affects: `add` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_f32_1k | 1024 | 4000 | 1.90 | 0.34 | 5.63 | 0.539 | 3.034 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_4k | 4096 | 4000 | 7.51 | 1.28 | 5.85 | 0.546 | 3.193 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_16k | 16384 | 4000 | 29.92 | 7.19 | 4.16 | 0.548 | 2.279 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_f32_64k | 65536 | 1953 | 138.42 | 27.95 | 4.95 | 0.473 | 2.345 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AddScalarBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float add kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_f32_1k | 1024 | 4000 | 1.95 | 0.26 | 7.42 | 0.524 | 3.889 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_4k | 4096 | 4000 | 7.81 | 1.05 | 7.41 | 0.524 | 3.886 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_16k | 16384 | 4000 | 31.26 | 4.27 | 7.32 | 0.524 | 3.835 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_scalar_kernel_f32_64k | 65536 | 1953 | 125.88 | 16.93 | 7.44 | 0.521 | 3.872 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastAddDispatch<float> scalar lhs

- Affects: `add` float scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_f32_1k | 1024 | 4000 | 2.40 | 0.34 | 7.15 | 0.426 | 3.045 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_4k | 4096 | 4000 | 9.40 | 1.12 | 8.37 | 0.436 | 3.645 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_16k | 16384 | 4000 | 37.72 | 4.45 | 8.48 | 0.434 | 3.685 | 0.00000000 | 0.00000000 | 0.0000000000 |
| add_bcast_f32_64k | 65536 | 1953 | 149.51 | 16.99 | 8.80 | 0.438 | 3.856 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Mul<float>

- Affects: `mul` float optimized path, non-broadcast dense tensors.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_f32_1k | 1024 | 4000 | 2.82 | 0.37 | 7.73 | 0.363 | 2.805 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_4k | 4096 | 4000 | 11.24 | 1.40 | 8.02 | 0.365 | 2.923 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_16k | 16384 | 4000 | 44.84 | 7.47 | 6.01 | 0.365 | 2.194 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_f32_64k | 65536 | 1953 | 179.29 | 25.00 | 7.17 | 0.366 | 2.621 | 0.00000000 | 0.00000000 | 0.0000000000 |

### MulSimpleBroadcast<float> direct kernel

- Affects: Direct scalar-broadcast float mul kernel used by the fivefold broadcast fast path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_f32_1k | 1024 | 4000 | 2.37 | 0.29 | 8.14 | 0.433 | 3.520 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_4k | 4096 | 4000 | 9.33 | 1.17 | 7.95 | 0.439 | 3.490 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_16k | 16384 | 4000 | 37.43 | 4.70 | 7.96 | 0.438 | 3.486 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_scalar_kernel_f32_64k | 65536 | 1953 | 149.27 | 18.70 | 7.98 | 0.439 | 3.505 | 0.00000000 | 0.00000000 | 0.0000000000 |

### BroadcastMulDispatch<float> scalar lhs

- Affects: `mul` float scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_f32_1k | 1024 | 4000 | 2.41 | 0.35 | 6.82 | 0.426 | 2.904 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_4k | 4096 | 4000 | 9.44 | 1.23 | 7.70 | 0.434 | 3.339 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_16k | 16384 | 4000 | 37.75 | 4.87 | 7.74 | 0.434 | 3.361 | 0.00000000 | 0.00000000 | 0.0000000000 |
| mul_bcast_f32_64k | 65536 | 1953 | 150.74 | 18.87 | 7.99 | 0.435 | 3.472 | 0.00000000 | 0.00000000 | 0.0000000000 |

### SubWithActivation<float>

- Affects: `sub` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_f32_1k | 1024 | 4000 | 1.89 | 0.34 | 5.57 | 0.543 | 3.022 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_4k | 4096 | 4000 | 7.50 | 1.28 | 5.87 | 0.546 | 3.204 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_16k | 16384 | 4000 | 30.00 | 7.36 | 4.08 | 0.546 | 2.226 | 0.00000000 | 0.00000000 | 0.0000000000 |
| sub_f32_64k | 65536 | 1953 | 119.59 | 27.41 | 4.36 | 0.548 | 2.391 | 0.00000000 | 0.00000000 | 0.0000000000 |

### Div<float>

- Affects: `div` float non-broadcast optimized path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| div_f32_1k | 1024 | 4000 | 4.71 | 2.35 | 2.01 | 0.218 | 0.437 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_4k | 4096 | 4000 | 18.68 | 9.32 | 2.00 | 0.219 | 0.440 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_16k | 16384 | 4000 | 74.79 | 37.40 | 2.00 | 0.219 | 0.438 | 0.00000000 | 0.00000000 | 0.0000000000 |
| div_f32_64k | 65536 | 1953 | 298.80 | 149.26 | 2.00 | 0.219 | 0.439 | 0.00000000 | 0.00000000 | 0.0000000000 |

### AffineQuantize<int8>

- Affects: `quantize` float-to-int8 path used by int8 activations and weights.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i8_1k | 1024 | 4000 | 14.78 | 1.12 | 13.15 | 0.069 | 0.911 | 0 | 0 |
| affine_q_i8_4k | 4096 | 4000 | 79.23 | 4.46 | 17.78 | 0.052 | 0.919 | 0 | 0 |
| affine_q_i8_16k | 16384 | 4000 | 332.33 | 17.78 | 18.69 | 0.049 | 0.922 | 0 | 0 |
| affine_q_i8_64k | 65536 | 1953 | 1324.34 | 70.92 | 18.67 | 0.049 | 0.924 | 0 | 0 |

### AffineQuantize<uint8>

- Affects: `quantize` float-to-uint8 path used by uint8 activations.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_u8_1k | 1024 | 4000 | 8.92 | 1.40 | 6.38 | 0.115 | 0.732 | 0 | 0 |
| affine_q_u8_4k | 4096 | 4000 | 35.64 | 5.62 | 6.35 | 0.115 | 0.729 | 0 | 0 |
| affine_q_u8_16k | 16384 | 4000 | 142.95 | 22.49 | 6.35 | 0.115 | 0.728 | 0 | 0 |
| affine_q_u8_64k | 65536 | 1953 | 570.00 | 89.59 | 6.36 | 0.115 | 0.731 | 0 | 0 |

### AffineQuantize<int16>

- Affects: `quantize` float-to-int16 path used by q15-style operator staging.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine_q_i16_1k | 1024 | 4000 | 8.90 | 1.01 | 8.78 | 0.115 | 1.010 | 0 | 0 |
| affine_q_i16_4k | 4096 | 4000 | 35.47 | 4.01 | 8.86 | 0.115 | 1.023 | 0 | 0 |
| affine_q_i16_16k | 16384 | 4000 | 142.08 | 15.91 | 8.93 | 0.115 | 1.030 | 0 | 0 |
| affine_q_i16_64k | 65536 | 1953 | 567.25 | 63.61 | 8.92 | 0.116 | 1.030 | 0 | 0 |

### AveragePool<uint8>

- Affects: `average_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 121.52 | 77.68 | 1.56 | 0.539 | 0.844 | 0 | 0 |
| avgpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1111.82 | 855.57 | 1.30 | 0.755 | 0.982 | 0 | 0 |
| avgpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 435.08 | 337.98 | 1.29 | 0.763 | 0.982 | 0 | 0 |

### MaxPool<uint8>

- Affects: `max_pool` quantized depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_u8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 214.40 | 23.62 | 9.08 | 0.306 | 2.775 | 0 | 0 |
| maxpool_u8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2127.89 | 237.76 | 8.95 | 0.395 | 3.532 | 0 | 0 |
| maxpool_u8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 816.82 | 93.44 | 8.74 | 0.406 | 3.551 | 0 | 0 |

### HardSwish<uint8>

- Affects: `hard_swish` quantized uint8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_u8_1k | 1024 | 4000 | 11.42 | 7.76 | 1.47 | 0.090 | 0.132 | 0 | 0 |
| hardswish_u8_4k | 4096 | 4000 | 51.10 | 30.96 | 1.65 | 0.080 | 0.132 | 0 | 0 |
| hardswish_u8_16k | 16384 | 4000 | 211.64 | 123.62 | 1.71 | 0.077 | 0.133 | 0 | 0 |
| hardswish_u8_64k | 65536 | 1953 | 848.37 | 494.51 | 1.72 | 0.077 | 0.133 | 0 | 0 |

### HardSwish<int8>

- Affects: `hard_swish` quantized int8 fixed-point multiplier path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardswish_i8_1k | 1024 | 4000 | 10.66 | 7.04 | 1.51 | 0.096 | 0.146 | 0 | 0 |
| hardswish_i8_4k | 4096 | 4000 | 48.99 | 28.03 | 1.75 | 0.084 | 0.146 | 0 | 0 |
| hardswish_i8_16k | 16384 | 4000 | 202.69 | 111.97 | 1.81 | 0.081 | 0.146 | 0 | 0 |
| hardswish_i8_64k | 65536 | 1953 | 814.10 | 448.04 | 1.82 | 0.081 | 0.146 | 0 | 0 |

### ArgMin<float>

- Affects: `arg_min` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmin_f32_1k | 1024 | 4000 | 0.95 | 0.91 | 1.05 | 1.078 | 1.128 | 0 | 0 |
| argmin_f32_16k | 16384 | 4000 | 14.99 | 13.05 | 1.15 | 1.093 | 1.255 | 0 | 0 |
| argmin_f32_64k | 65536 | 1953 | 59.72 | 51.22 | 1.17 | 1.097 | 1.279 | 0 | 0 |

### ArgMax<float>

- Affects: `arg_max` float last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_f32_1k | 1024 | 4000 | 0.95 | 0.91 | 1.04 | 1.081 | 1.128 | 0 | 0 |
| argmax_f32_16k | 16384 | 4000 | 15.00 | 13.05 | 1.15 | 1.092 | 1.256 | 0 | 0 |
| argmax_f32_64k | 65536 | 1953 | 59.71 | 51.22 | 1.17 | 1.098 | 1.280 | 0 | 0 |

### ArgMax<int8>

- Affects: `arg_max` int8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_i8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.68 | 0.728 | 5.588 | 0 | 0 |
| argmax_i8_16k | 16384 | 4000 | 22.41 | 1.61 | 13.94 | 0.731 | 10.192 | 0 | 0 |
| argmax_i8_64k | 65536 | 1953 | 89.72 | 6.46 | 13.89 | 0.730 | 10.149 | 0 | 0 |

### ArgMax<uint8>

- Affects: `arg_max` uint8 last-axis reduction, including tie-keep-first semantics.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_u8_1k | 1024 | 4000 | 1.41 | 0.18 | 7.67 | 0.728 | 5.577 | 0 | 0 |
| argmax_u8_16k | 16384 | 4000 | 22.39 | 1.61 | 13.95 | 0.732 | 10.206 | 0 | 0 |
| argmax_u8_64k | 65536 | 1953 | 89.63 | 6.40 | 14.00 | 0.731 | 10.235 | 0 | 0 |

## `tflite/kernels/internal/optimized/reduce.h`

- Summary: Follow-up P1 closure for the uint8 keep-dims height-width reduction path and the float last-dimension mean specialization.

### MeanImpl<uint8>

- Affects: `reduce.h` 4D keep-dims height-width reduction path for uint8.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_u8_8x8x48 | B1 I8x8x48 A(1,2) | 4000 | 3.47 | 2.25 | 1.54 | 0.886 | 1.365 | 0 | 0 |
| mean_u8_16x16x96 | B1 I16x16x96 A(1,2) | 4000 | 18.85 | 14.95 | 1.26 | 1.304 | 1.644 | 0 | 0 |
| mean_u8_32x8x160 | B1 I32x8x160 A(1,2) | 3125 | 39.86 | 24.79 | 1.61 | 1.028 | 1.652 | 0 | 0 |

### Mean<float> last-dim

- Affects: `reduce.h` float specialization that reduces only the last dimension.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_f32_lastdim_256x48 | R256 C48 axis(1) | 4000 | 13.69 | 13.62 | 1.00 | 0.898 | 0.902 | 0.00000024 | 0.00003184 | 0.0000000346 |
| mean_f32_lastdim_256x192 | R256 C192 axis(1) | 2604 | 69.08 | 37.44 | 1.85 | 0.712 | 1.313 | 0.00000021 | 0.00029705 | 0.0000000323 |
| mean_f32_lastdim_128x640 | R128 C640 axis(1) | 1562 | 113.03 | 60.39 | 1.87 | 0.725 | 1.357 | 0.00000018 | 0.00002661 | 0.0000000364 |

## `tflite/kernels/internal/optimized/resize_bilinear.h`

- Summary: Follow-up P1 closure for the generic float bilinear kernel and the uint8 generic-small-channel interpolation path.

### ResizeBilinear<float>

- Affects: `resize_bilinear.h` generic float bilinear accumulation kernel.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize_f32_17x19x48_to_29x31 | B1 I17x19x48 O29x31 AC0 HPC0 | 423 | 241.17 | 241.37 | 1.00 | 1.252 | 1.251 | 0.00000000 | 0.00000000 | 0.0000000000 |
| resize_f32_23x27x96_to_37x41 | B1 I23x27x96 O37x41 AC0 HPC1 | 125 | 787.32 | 787.41 | 1.00 | 1.295 | 1.295 | 0.00000000 | 0.00000000 | 0.0000000000 |
| resize_f32_15x21x160_to_28x35 | B1 I15x21x160 O28x35 AC1 HPC0 | 116 | 851.86 | 852.64 | 1.00 | 1.288 | 1.287 | 0.00000000 | 0.00000000 | 0.0000000000 |

### ResizeBilinear<uint8>

- Affects: `resize_bilinear.h` uint8 generic-small-channel interpolation path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resize_u8_17x19x48_to_29x31 | B1 I17x19x48 O29x31 AC0 HPC0 | 423 | 287.94 | 140.66 | 2.05 | 1.049 | 2.148 | 0 | 0 |
| resize_u8_23x27x96_to_37x41 | B1 I23x27x96 O37x41 AC0 HPC1 | 125 | 947.90 | 440.16 | 2.15 | 1.075 | 2.316 | 0 | 0 |
| resize_u8_15x21x160_to_28x35 | B1 I15x21x160 O28x35 AC1 HPC0 | 116 | 1010.75 | 458.35 | 2.21 | 1.086 | 2.395 | 0 | 0 |

## `tflite/kernels/fully_connected.cc`

- Summary: Dense float FullyConnected on RVV now takes the `EvalPie` route, so this split benchmark keeps the measurement alongside the source-file section and exercises the same matvec-style accumulation shape.

### FullyConnected<float> dense EvalPie route

- Affects: `fully_connected.cc` dense float RVV dispatch that now routes through `EvalPie`; this benchmark mirrors that matrix-batch-vector accumulation path inside the split operator harness.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fc_dense_small_b4 | B4 I64 O64 | 3906 | 24.33 | 21.22 | 1.15 | 1.347 | 1.544 | 0.00000000 | 0.00000000 | 0.0000000000 |
| fc_dense_mid_b4 | B4 I128 O128 | 976 | 87.02 | 87.08 | 1.00 | 1.506 | 1.505 | 0.00000000 | 0.00000000 | 0.0000000000 |
| fc_dense_large_b4 | B4 I2048 O640 | 20 | 7214.61 | 7218.26 | 1.00 | 1.453 | 1.453 | 0.00000000 | 0.00000000 | 0.0000000000 |

## `tflite/kernels/lstm_eval.cc`

- Summary: Float gate and output/projection paths now reuse the RVV-enabled `tensor_utils` matvec helpers, so this section keeps the operator-level numbers separate from generic helper benchmarks.

### LstmGate<float>

- Affects: `lstm_eval` float gate path (`input_to_gate` + `recurrent_to_gate` + sigmoid).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_gate_small_b1 | B1 I128 O128 C512 | 488 | 189.61 | 189.46 | 1.00 | 1.383 | 1.384 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_gate_small_b4 | B4 I128 O128 C512 | 122 | 758.13 | 752.68 | 1.01 | 1.383 | 1.393 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_gate_large_b1 | B1 I256 O256 C1024 | 122 | 763.22 | 763.22 | 1.00 | 1.374 | 1.374 | 0.00000000 | 0.00000000 | 0.0000000000 |

### LstmOutput<float>

- Affects: `lstm_eval` float output/projection path (`tanh(cell)` + output gate + projection).

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Max rel diff | Mean abs diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm_output_small_b1 | B1 C512 O128 | 972 | 108.15 | 107.75 | 1.00 | 1.217 | 1.221 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_output_small_b4 | B4 C512 O128 | 243 | 430.74 | 430.33 | 1.00 | 1.222 | 1.223 | 0.00000000 | 0.00000000 | 0.0000000000 |
| lstm_output_large_b1 | B1 C1024 O256 | 243 | 394.36 | 394.38 | 1.00 | 1.332 | 1.332 | 0.00000000 | 0.00000000 | 0.0000000000 |

## `tflite/kernels/internal/optimized/integer_ops/add.h`

- Summary: Recent RVV work added both int8/int16 elementwise add coverage and the scalar-broadcast kernels used by the fivefold broadcast fast path.

### Add<int8>

- Affects: `integer_ops/add.h` int8 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i8_1k | 1024 | 4000 | 10.13 | 5.90 | 1.72 | 0.101 | 0.174 | 0 | 0 |
| add_i8_4k | 4096 | 4000 | 50.75 | 23.56 | 2.15 | 0.081 | 0.174 | 0 | 0 |
| add_i8_16k | 16384 | 4000 | 209.98 | 94.30 | 2.23 | 0.078 | 0.174 | 0 | 0 |
| add_i8_64k | 65536 | 1953 | 839.69 | 377.02 | 2.23 | 0.078 | 0.174 | 0 | 0 |

### AddScalarBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast add kernel used by the fivefold broadcast fast path in `integer_ops/add.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_scalar_kernel_i8_1k | 1024 | 4000 | 6.89 | 4.16 | 1.65 | 0.149 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_4k | 4096 | 4000 | 33.82 | 16.69 | 2.03 | 0.121 | 0.245 | 0 | 0 |
| add_scalar_kernel_i8_16k | 16384 | 4000 | 130.10 | 66.67 | 1.95 | 0.126 | 0.246 | 0 | 0 |
| add_scalar_kernel_i8_64k | 65536 | 1953 | 506.98 | 266.83 | 1.90 | 0.129 | 0.246 | 0 | 0 |

### BroadcastAddDispatch<int8> scalar lhs

- Affects: `integer_ops/add.h` int8 scalar-broadcast fast path via `BroadcastAddDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_bcast_i8_1k | 1024 | 4000 | 14.48 | 14.42 | 1.00 | 0.071 | 0.071 | 0 | 0 |
| add_bcast_i8_4k | 4096 | 4000 | 67.45 | 67.45 | 1.00 | 0.061 | 0.061 | 0 | 0 |
| add_bcast_i8_16k | 16384 | 4000 | 271.49 | 271.37 | 1.00 | 0.060 | 0.060 | 0 | 0 |
| add_bcast_i8_64k | 65536 | 1953 | 1055.12 | 1055.58 | 1.00 | 0.062 | 0.062 | 0 | 0 |

### Add<int16>

- Affects: `integer_ops/add.h` int16 elementwise quantized add path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add_i16_1k | 1024 | 4000 | 10.26 | 5.79 | 1.77 | 0.100 | 0.177 | 0 | 0 |
| add_i16_4k | 4096 | 4000 | 50.06 | 23.10 | 2.17 | 0.082 | 0.177 | 0 | 0 |
| add_i16_16k | 16384 | 4000 | 207.68 | 92.48 | 2.25 | 0.079 | 0.177 | 0 | 0 |
| add_i16_64k | 65536 | 1953 | 833.33 | 369.55 | 2.25 | 0.079 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mul.h`

- Summary: Recent RVV work added both int8 elementwise mul coverage and the scalar-broadcast kernels used by the quantized broadcast fast path.

### Mul<int8>

- Affects: `integer_ops/mul.h` int8 elementwise quantized mul path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_i8_1k | 1024 | 4000 | 10.86 | 3.23 | 3.37 | 0.094 | 0.317 | 0 | 0 |
| mul_i8_4k | 4096 | 4000 | 48.82 | 12.85 | 3.80 | 0.084 | 0.319 | 0 | 0 |
| mul_i8_16k | 16384 | 4000 | 197.74 | 51.36 | 3.85 | 0.083 | 0.319 | 0 | 0 |
| mul_i8_64k | 65536 | 1953 | 793.71 | 205.30 | 3.87 | 0.083 | 0.319 | 0 | 0 |

### MulSimpleBroadcast<int8> direct kernel

- Affects: Direct int8 scalar-broadcast mul kernel used by the fivefold broadcast fast path in `integer_ops/mul.h`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_scalar_kernel_i8_1k | 1024 | 4000 | 9.92 | 3.29 | 3.02 | 0.103 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_4k | 4096 | 4000 | 45.54 | 13.19 | 3.45 | 0.090 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_16k | 16384 | 4000 | 187.00 | 52.74 | 3.55 | 0.088 | 0.311 | 0 | 0 |
| mul_scalar_kernel_i8_64k | 65536 | 1953 | 751.82 | 210.88 | 3.57 | 0.087 | 0.311 | 0 | 0 |

### BroadcastMulDispatch<int8> scalar lhs

- Affects: `integer_ops/mul.h` int8 scalar-broadcast fast path via `BroadcastMulDispatch`.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mul_bcast_i8_1k | 1024 | 4000 | 10.60 | 10.52 | 1.01 | 0.097 | 0.097 | 0 | 0 |
| mul_bcast_i8_4k | 4096 | 4000 | 48.01 | 48.01 | 1.00 | 0.085 | 0.085 | 0 | 0 |
| mul_bcast_i8_16k | 16384 | 4000 | 195.98 | 196.06 | 1.00 | 0.084 | 0.084 | 0 | 0 |
| mul_bcast_i8_64k | 65536 | 1953 | 787.96 | 787.78 | 1.00 | 0.083 | 0.083 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/sub.h`

- Summary: This section isolates the int16 quantized subtract path that gained a dedicated RVV elementwise implementation.

### Sub<int16>

- Affects: `integer_ops/sub.h` int16 elementwise quantized sub path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_i16_1k | 1024 | 4000 | 10.07 | 5.79 | 1.74 | 0.102 | 0.177 | 0 | 0 |
| sub_i16_4k | 4096 | 4000 | 50.64 | 23.11 | 2.19 | 0.081 | 0.177 | 0 | 0 |
| sub_i16_16k | 16384 | 4000 | 207.16 | 92.42 | 2.24 | 0.079 | 0.177 | 0 | 0 |
| sub_i16_64k | 65536 | 1953 | 833.73 | 369.70 | 2.26 | 0.079 | 0.177 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/pooling.h`

- Summary: This section keeps the signed int8 pooling kernels separate from the uint8 `optimized_ops.h` pooling paths, because their RVV logic and rounding semantics are different.

### AveragePool<int8>

- Affects: `average_pool` int8 depth inner loop, signed round-away-from-zero, and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| avgpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 182.46 | 64.74 | 2.82 | 0.359 | 1.012 | 0 | 0 |
| avgpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 1426.87 | 685.05 | 2.08 | 0.589 | 1.226 | 0 | 0 |
| avgpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 566.55 | 271.57 | 2.09 | 0.586 | 1.222 | 0 | 0 |

### MaxPool<int8>

- Affects: `max_pool` int8 depth inner loop and clamp-store path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maxpool_i8_32x32x64 | B1 I32x32x64 F2x2 S2x2 | 1953 | 227.41 | 24.20 | 9.40 | 0.288 | 2.708 | 0 | 0 |
| maxpool_i8_56x56x128 | B1 I56x56x128 F3x3 S2x2 | 152 | 2220.10 | 239.31 | 9.28 | 0.378 | 3.509 | 0 | 0 |
| maxpool_i8_14x14x256 | B1 I14x14x256 F3x3 S1x1 | 385 | 856.38 | 94.90 | 9.02 | 0.387 | 3.496 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h`

- Summary: This section tracks the RVV general depthwise int8 kernels, including the fixed-depth and fixed-depth-multiplier specializations added to mirror the NEON dispatch table.

### DepthwiseConv<int8>

- Affects: `depthwise_conv` int8 general path, including fixed input-depth and depth-multiplier RVV kernels aligned with NEON dispatch.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dwconv_i8_dm1_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM1 | 1975 | 185.88 | 115.38 | 1.61 | 0.349 | 0.562 | 0 | 0 |
| dwconv_i8_dm1_d8_s1 | B1 I32x32x8 F3x3 S1x1 DM1 | 987 | 342.46 | 152.17 | 2.25 | 0.378 | 0.852 | 0 | 0 |
| dwconv_i8_dm1_d12_s1 | B1 I32x32x12 F3x3 S1x1 DM1 | 658 | 509.60 | 195.37 | 2.61 | 0.381 | 0.995 | 0 | 0 |
| dwconv_i8_dm1_d16_s2 | B1 I32x32x16 F3x3 S2x2 DM1 | 1975 | 171.37 | 71.86 | 2.38 | 0.378 | 0.902 | 0 | 0 |
| dwconv_i8_dm2_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM2 | 987 | 306.12 | 193.37 | 1.58 | 0.423 | 0.670 | 0 | 0 |
| dwconv_i8_dm2_d8_s2 | B1 I32x32x8 F3x3 S2x2 DM2 | 1975 | 153.08 | 96.83 | 1.58 | 0.423 | 0.669 | 0 | 0 |
| dwconv_i8_dm4_d1_s1 | B1 I32x32x1 F3x3 S1x1 DM4 | 1975 | 130.37 | 107.75 | 1.21 | 0.497 | 0.601 | 0 | 0 |
| dwconv_i8_dm4_d4_s1 | B1 I32x32x4 F3x3 S1x1 DM4 | 493 | 452.86 | 385.57 | 1.17 | 0.572 | 0.672 | 0 | 0 |
| dwconv_i8_dm8_d2_s1 | B1 I32x32x2 F3x3 S1x1 DM8 | 493 | 391.44 | 282.16 | 1.39 | 0.662 | 0.919 | 0 | 0 |
| dwconv_i8_dm8_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM8 | 3950 | 53.77 | 44.39 | 1.21 | 0.603 | 0.730 | 0 | 0 |
| dwconv_i8_dm16_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM16 | 1975 | 95.58 | 64.65 | 1.48 | 0.678 | 1.002 | 0 | 0 |
| dwconv_i8_dm20_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM20 | 1580 | 116.89 | 96.92 | 1.21 | 0.693 | 0.836 | 0 | 0 |
| dwconv_i8_dm32_d1_s2 | B1 I32x32x1 F3x3 S2x2 DM32 | 987 | 176.87 | 129.54 | 1.37 | 0.733 | 1.000 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/depthwise_conv_hybrid.h`

- Note: The recent change in this file only reroutes hybrid dispatch to reuse the RVV depthwise kernels above, so it intentionally shares the same numeric benchmark coverage.

## `tflite/kernels/internal/optimized/integer_ops/leaky_relu.h`

- Summary: This section isolates the int16 quantized LeakyReLU path that received a dedicated RVV implementation.

### LeakyRelu<int16>

- Affects: `integer_ops/leaky_relu.h` int16 quantized leaky-relu path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| leaky_relu_i16_1k | 1024 | 4000 | 13.36 | 4.08 | 3.28 | 0.077 | 0.251 | 0 | 0 |
| leaky_relu_i16_4k | 4096 | 4000 | 54.04 | 16.35 | 3.30 | 0.076 | 0.250 | 0 | 0 |
| leaky_relu_i16_16k | 16384 | 4000 | 215.11 | 65.31 | 3.29 | 0.076 | 0.251 | 0 | 0 |
| leaky_relu_i16_64k | 65536 | 1953 | 862.04 | 261.33 | 3.30 | 0.076 | 0.251 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/lut.h`

- Summary: This section keeps the uint8 and int8 lookup-table kernels separate because they were vectorized in the same commit but have different input domains.

### LookupTable<uint8>

- Affects: `integer_ops/lut.h` uint8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_u8_1k | 1024 | 4000 | 1.41 | 0.85 | 1.66 | 0.729 | 1.207 | 0 | 0 |
| lut_u8_4k | 4096 | 4000 | 5.61 | 3.40 | 1.65 | 0.730 | 1.205 | 0 | 0 |
| lut_u8_16k | 16384 | 4000 | 22.41 | 13.57 | 1.65 | 0.731 | 1.207 | 0 | 0 |
| lut_u8_64k | 65536 | 1953 | 89.60 | 54.30 | 1.65 | 0.731 | 1.207 | 0 | 0 |

### LookupTable<int8>

- Affects: `integer_ops/lut.h` int8 lookup-table path.

| Case | Vector size | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lut_i8_1k | 1024 | 4000 | 0.94 | 0.85 | 1.11 | 1.087 | 1.205 | 0 | 0 |
| lut_i8_4k | 4096 | 4000 | 3.74 | 3.39 | 1.10 | 1.095 | 1.207 | 0 | 0 |
| lut_i8_16k | 16384 | 4000 | 14.93 | 13.57 | 1.10 | 1.097 | 1.208 | 0 | 0 |
| lut_i8_64k | 65536 | 1953 | 60.00 | 54.17 | 1.11 | 1.092 | 1.210 | 0 | 0 |

## `tflite/kernels/internal/optimized/integer_ops/mean.h`

- Summary: This section tracks the int8 height-width reduction path that gained RVV coverage in the latest integer-ops commit.

### Mean<int8>

- Affects: `integer_ops/mean.h` int8 height-width reduction path.

| Case | Shape | Iterations | Scalar us | RVV us | Speedup | Scalar GOPS | RVV GOPS | Max abs diff | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_i8_8x8x64 | B1 I8x8x64 A(1,2) | 4000 | 6.17 | 1.63 | 3.78 | 0.664 | 2.508 | 0 | 0 |
| mean_i8_16x16x128 | B1 I16x16x128 A(1,2) | 3906 | 38.49 | 11.61 | 3.32 | 0.851 | 2.823 | 0 | 0 |
| mean_i8_32x32x256 | B1 I32x32x256 A(1,2) | 488 | 1458.37 | 92.18 | 15.82 | 0.180 | 2.844 | 0 | 0 |

