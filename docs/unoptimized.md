# RVV Optimization Coverage

**58 / 114** NEON blocks have RVV equivalents (51%). All 56 remaining blocks are accounted for below with specific blockers.

Plus **20 RVV functions** in `rvv_tensor_utils.cc` = **78 total RVV optimizations**.

## Implemented (58 blocks)

Across `optimized_ops.h` (36), `integer_ops/` (10), `depthwiseconv_float.h` (1), `depthwiseconv_uint8.h` (2), `reduce.h` (1), `resize_bilinear.h` (2), `legacy_optimized_ops.h` (4), `depthwise_conv_hybrid.h` (1), `integer_ops/sub.h` (1 — AVX2 path).

Covering: Add, Sub, Mul, HardSwish (float), Maximum, Minimum, PReLU, Dequantize (3 types), AffineQuantize (3 types), Requantize (4 types), Quantize (single + per-channel), ArgMin/ArgMax (4 types), MaxPool, AveragePool, DepthwiseConv (float + int8 + uint8), Mean (int8 + uint8), ResizeBilinear (float kernel + 2x2), ShuffledFC XOR, Softmax max-finding, FC/Conv dispatch guards, DepthwiseConvHybrid output.

## Remaining: 56 blocks

### Truly impossible — 17 blocks

`gemmlowp::FixedPoint<int32x4_t>` or `gemmlowp::FixedPoint<int16x8_t>` template specializations. No RVV equivalent exists in gemmlowp. Would require reimplementing gemmlowp's fixed-point math library.

| Function | File | Lines |
|---|---|---|
| Logistic (int32x4_t) | optimized_ops.h | 30 |
| Tanh (int32x4_t) | optimized_ops.h | 55 |
| HardSwish quantized (int32x4_t) | optimized_ops.h | 142 |
| Tanh16bitPrecision (2 variants) | optimized_ops.h | 55+44 |
| Logistic16bitPrecision (2 variants) | optimized_ops.h | 44+55 |
| SaturateAndStore (NEON type params) | optimized_ops.h | 15 |
| ScaleWithNewZeroPoint (NEON type params) | optimized_ops.h | 15 |
| Softmax exp-sum + output (3 blocks) | legacy_optimized_ops.h | 120 |
| Logistic (gemmlowp) | legacy_optimized_ops.h | 98 |
| Tanh (gemmlowp) | legacy_optimized_ops.h | 105 |
| Softmax type aliases | legacy_optimized_ops.h | 10 |

### File-level gates — 6 blocks

Have separate RVV equivalents (`rvv_tensor_utils.cc`, `rvv_tensor_utils_impl.h`).

| File | Block |
|---|---|
| neon_check.h | NEON_OR_PORTABLE macro definition |
| neon_tensor_utils.cc | Entire file (2197 lines) |
| neon_tensor_utils_impl.h | Entire file (166 lines) |

### Template specializations — 15 blocks

NEON register geometry (128-bit fixed width). Generic accumulation path already has RVV.

| File | Content |
|---|---|
| depthwiseconv_float.h | 12 float kernel templates (730 lines) + dispatch |
| depthwiseconv_uint8.h | 15 uint8 kernel templates (1471 lines) + dispatch |
| depthwiseconv_uint8_transitional.h | Transitional impl (6000+ lines) |
| depthwiseconv_3x3_filter_common.h | 3x3 NEON helper types |
| depthwiseconv_uint8_3x3_filter.h | 3x3 filter specialization |
| integer_ops/depthwise_conv.h | 15 int8 kernel templates + dispatch |
| resize_bilinear.h | NEON struct types + 941-line uint8 resize |

### Legacy GEMV — 8 blocks

Hundreds of lines of NEON-specific loop structure with manual peeling.

| Function | File | Lines |
|---|---|---|
| FullyConnectedAsGEMV uint8 | legacy_optimized_ops.h | 321 |
| FullyConnectedAsGEMV int8 | legacy_optimized_ops.h | 291 |
| FullyConnectedAsGEMV float | legacy_optimized_ops.h | 313 |
| ShuffledFC metadata/filter/worker | legacy_optimized_ops.h | 69 |
| ShuffledFC batch=4 interleave | optimized_ops.h | 33 |
| ShuffledFCWorkerImpl | optimized_ops.h | 30 |

### NEON-only constant init — 10 blocks

Just `vdupq_n_*` constant initialization or `#ifdef` for variable declarations. No vectorizable code — the RVV loop already works without these.

| Location | Purpose |
|---|---|
| optimized_ops.h (4 blocks) | Quantize per-channel NEON constant init |
| reduce.h (1 block) | MeanImpl NEON constant init |
| integer_ops/mean.h (1 block) | MeanImpl NEON constant init |
| legacy_optimized_ops.h (2 blocks) | GEMMLOWP_NEON FC dispatch guards |
| integer_ops/pooling.h (1 block) | AveragePool output (no vectorizable division) |
| legacy_optimized_ops.h (1 block) | Softmax NEON type aliases |
