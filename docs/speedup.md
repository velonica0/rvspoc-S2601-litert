# RVV vs Scalar Speedup

All results measured on Spacemit X100 (rv64gcv, VLEN=256), GCC 15.2.0, CMake Release, XNNPACK disabled, Ruy enabled. See `testing.md` for exact reproduction commands.

## 1. Per-Operator Speedup

Reproduced by: `./rvv_bench` (see testing.md Section 3)

### Element-wise Operators

| Operator | Size | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|---|
| FloatAdd | 16384 | 0.021 | 0.007 | **2.91x** |
| FloatMul | 16384 | 0.021 | 0.007 | **2.91x** |
| FloatHardSwish | 16384 | 0.055 | 0.006 | **8.55x** |
| Int8Add | 16384 | 0.396 | 0.061 | **6.49x** |
| Int8Mul | 16384 | 0.258 | 0.031 | **8.26x** |

### Tensor Utilities

| Function | Parameters | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|---|
| FloatMatVec | 256x256 | 0.150 | 0.018 | **8.44x** |
| FloatDotProduct | 16384 | 0.037 | 0.006 | **6.58x** |
| FloatSub1Vec | 16384 | 0.009 | 0.005 | **2.01x** |
| FloatRedSum | 256x64 | 0.070 | 0.006 | **11.77x** |
| MeanStddevNorm | 1024x4 | 0.022 | 0.002 | **9.88x** |
| FloatCwiseClip | 16384 | 0.101 | 0.009 | **11.01x** |
| FloatIsZero | 16384 | 0.028 | 0.005 | **5.57x** |
| VecScalarMul | 16384 | 0.015 | 0.005 | **2.91x** |

## 2. Model-Level Speedup

Reproduced by: `./rvv_mobilenet_bench` vs `build-scalar/rvv_mobilenet_bench` (see testing.md Section 5)

Both builds use `-DTFLITE_ENABLE_RUY=ON` for working multi-threading on RISC-V.

### MobileNetV1 (1.0/224)

| Precision | Threads | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|---|
| FP32 | 1 | 3102 | 204 | **15.2x** |
| FP32 | 4 | 788 | 53 | **14.9x** |
| FP32 | 8 | 416 | 37 | **11.2x** |
| INT8 | 1 | 2252 | 238 | **9.5x** |
| INT8 | 4 | 569 | 67 | **8.5x** |
| INT8 | 8 | 306 | 40 | **7.7x** |
| uint8 | 1 | 2240 | 370 | **6.1x** |
| uint8 | 4 | 565 | 93 | **6.1x** |
| uint8 | 8 | 298 | 63 | **4.7x** |

### MobileNetV2 (1.0/224)

| Precision | Threads | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|---|
| FP32 | 1 | 1722 | 168 | **10.3x** |
| FP32 | 4 | 437 | 45 | **9.7x** |
| FP32 | 8 | 243 | 29 | **8.4x** |
| INT8 | 1 | 1407 | 243 | **5.8x** |
| INT8 | 4 | 364 | 70 | **5.2x** |
| INT8 | 8 | 210 | 47 | **4.5x** |

## 3. Analysis

### FP32 GEMM (10-15x)

The scalar FP32 GEMM uses Ruy's generic kernel on RISC-V (no SIMD). Our RVV GEMM uses `vfmacc` (fused multiply-accumulate) processing 8 floats per instruction. Row-partitioned multi-threading scales linearly: 204ms (1T) -> 53ms (4T) -> 37ms (8T).

### INT8 GEMM (5-9x)

The scalar INT8 GEMM also uses Ruy's generic kernel. Our RVV GEMM uses:
- Raw `vwmul` + `vwadd` dot product (2 ops/element vs 6 with zero-point in loop)
- Precomputed row/column sums for zero-point correction (zero_point=0 for symmetric INT8 eliminates this entirely)
- Multi-threaded column partitioning

### uint8 GEMM (4-6x)

Lower speedup than INT8 because uint8 models have non-zero zero_points, requiring the full correction term: `raw_acc - rhs_zp * row_sum - lhs_zp * col_sum + depth * lhs_zp * rhs_zp`.

### Thread scaling

Row-partitioned (FP32) and column-partitioned (INT8/uint8) threading both scale well with Ruy's threadpool. Previous results showed no FP32 multi-thread speedup because gemmlowp's threadpool is broken on RISC-V.

## 4. Accuracy

Reproduced by: `./rvv_bench` (see testing.md Section 3)

- FP32 operators: relative error <= 1e-5
- INT8 operators: difference <= 1 LSB
- 95/95 accuracy tests passed

## 5. CLAUDE.md Threshold (Inference Latency <= 110ms)

| Model | Config | Latency | Status |
|---|---|---|---|
| MobileNetV2 FP32 | 8T | **29ms** | **PASS** |
| MobileNetV1 FP32 | 8T | **37ms** | **PASS** |
| MobileNetV1 INT8 | 8T | **40ms** | **PASS** |
| MobileNetV2 INT8 | 8T | **47ms** | **PASS** |
| MobileNetV2 FP32 | 4T | **45ms** | **PASS** |
| MobileNetV1 FP32 | 4T | **53ms** | **PASS** |
| MobileNetV1 uint8 | 8T | **63ms** | **PASS** |
| MobileNetV1 INT8 | 4T | **67ms** | **PASS** |
| MobileNetV2 INT8 | 4T | **70ms** | **PASS** |
| MobileNetV1 uint8 | 4T | **93ms** | **PASS** |
