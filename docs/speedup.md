# RVV vs Scalar Speedup

## Test Environment

- **CPU**: Spacemit X100 (rv64imafdcv, VLEN=256)
- **OS**: Linux 6.18.3+ (openkylin)
- **Compiler**: GCC 15.2.0 with `-march=rv64gcv -O2`
- **Build**: CMake Release, XNNPACK disabled
- **Benchmark**: `rvv_speedup_test.cc`, 2000 iterations per measurement, 5 warmup iterations

## Element-wise Operators

| Operator | Size | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|---|
| FloatAdd | 16384 | 0.021 | 0.007 | **2.91x** |
| FloatMul | 16384 | 0.021 | 0.007 | **2.91x** |
| FloatHardSwish | 16384 | 0.055 | 0.006 | **8.62x** |
| Int8Add | 16384 | 0.396 | 0.060 | **6.64x** |
| Int8Mul | 16384 | 0.268 | 0.031 | **8.56x** |
| Int8MaxPool | 8x8x128 | 0.001 | 0.001 | 0.68x* |

*MaxPool at sub-microsecond scale; timer noise dominates.

## Tensor Utilities

| Function | Parameters | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|---|
| FloatMatVec | 256x256 | 0.161 | 0.018 | **9.03x** |
| FloatDotProduct | 16384 | 0.040 | 0.006 | **7.09x** |
| FloatSub1Vec | 16384 | 0.009 | 0.003 | **3.05x** |
| FloatRedSum | 256x64 | 0.070 | 0.006 | **11.71x** |
| MeanStddevNorm | 1024x4 | 0.022 | 0.002 | **9.87x** |
| FloatCwiseClip | 16384 | 0.022 | 0.009 | **2.41x** |
| FloatIsZero | 16384 | 0.028 | 0.005 | **5.69x** |
| VecScalarMul | 16384 | 0.015 | 0.005 | **2.90x** |

## Analysis

### Compute-bound ops (6-12x speedup)

ReductionSum (11.7x), MeanStddevNorm (9.9x), FloatMatVec (9.0x), HardSwish (8.6x), Int8Mul (8.6x), FloatDotProduct (7.1x), Int8Add (6.6x).

These perform multiple arithmetic operations per element (multiply-accumulate, widening, reduction). RVV's `vfmacc`/`vfredusum`/`vwmul` process many elements per instruction, and the compute cost dominates memory access.

### Memory-bound ops (2-3x speedup)

FloatAdd (2.9x), FloatMul (2.9x), Sub1Vec (3.0x), VecScalarMul (2.9x), CwiseClip (2.4x).

These perform 1-2 simple operations per element. At VLEN=256 (8 floats per vector), the theoretical max is ~8x, but memory bandwidth limits the actual gain to ~3x. Larger VLEN would not help further.

### Reduction ops (5-12x speedup)

IsZero (5.7x), ReductionSum (11.7x), DotProduct (7.1x).

These benefit from RVV's `vredsum`/`vfredusum` which reduce a full vector to a scalar in hardware, replacing NEON's `AccumulateNeonLane` (4 additions) and the scalar accumulation loop.

## Accuracy

All operators verified within project thresholds:
- FP32 operators: relative error ≤ 1e-5
- INT8 operators: difference ≤ 1 LSB
- 96/96 accuracy tests passed on Spacemit X100
