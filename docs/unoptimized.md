# Unoptimized Operators (USE_NEON blocks without RVV)

78 remaining `#ifdef USE_NEON` blocks across 14 files have no corresponding `#elif defined(USE_RVV)` path. On RISC-V these fall back to scalar code.

## Tier 1 — Simple Element-wise (trivial to implement)

| Function | File | Line | RVV Mapping |
|---|---|---|---|
| `MaximumElementwise` | optimized_ops.h | 7383 | `vfmax` / `vmax` |
| `MaximumScalarBroadcast` | optimized_ops.h | 7406 | `vfmax_vf` / `vmax_vx` |
| `MinimumElementwise` | optimized_ops.h | 7427 | `vfmin` / `vmin` |
| `MinimumScalarBroadcast` | optimized_ops.h | 7450 | `vfmin_vf` / `vmin_vx` |
| `PReluElementWise` | optimized_ops.h | 7598 | `vfmul` + `vmerge` on sign mask |
| `PReluScalarBroadcast` | optimized_ops.h | 7549 | `vfmul_vf` + `vmerge` |
| `AddScalarBroadcast(float)` | optimized_ops.h | 1687 | `vfadd_vf` + clamp |
| `AddScalarBroadcast(uint8)` | optimized_ops.h | 1781 | widen + quantized add + narrow |
| `MulElementwise(uint8)` | optimized_ops.h | 2189 | widen + `vwmul` + quantize |
| `MulSimpleBroadcast(uint8)` | optimized_ops.h | 2267 | widen + `vwmul_vx` + quantize |
| `MulSimpleBroadcast(float)` | optimized_ops.h | 2330 | `vfmul_vf` + clamp |
| `FindMaxValue` | optimized_ops.h | 3782 | `vredmaxu` |

## Tier 2 — Quantization / Dequantization (used by every quantized model)

| Function | File | Line | RVV Mapping |
|---|---|---|---|
| `Quantize(int32→int8, single multiplier)` | optimized_ops.h | 5237 | SRDH + narrow |
| `Quantize(int32→int8, per-channel)` | optimized_ops.h | 5385 | per-channel SRDH + narrow |
| `Quantize(int32→int16, per-channel)` | optimized_ops.h | 5475 | per-channel SRDH + narrow |
| `Quantize(int32→int8, double rounding)` | optimized_ops.h | 5524 | double-rounding variant |
| `Quantize(int32→int16, double rounding)` | optimized_ops.h | 5615 | double-rounding variant |
| `Requantize(int8→uint8)` | optimized_ops.h | 5877 | add 128 + reinterpret |
| `Requantize(uint8→int8)` | optimized_ops.h | 5964 | sub 128 + reinterpret |
| `Requantize(int8→int16)` | optimized_ops.h | 6042 | `vsext_vf2` |
| `Requantize(int16→int8)` | optimized_ops.h | 6119 | `vnclip` |
| `Dequantize(int8→float)` | optimized_ops.h | 6528 | `vsext` + `vfcvt` + `vfmul` |
| `Dequantize(uint8→float)` | optimized_ops.h | 6568 | `vzext` + `vfcvt` + `vfmacc` |
| `Dequantize(int16→float)` | optimized_ops.h | 6607 | `vsext` + `vfcvt` + `vfmul` |
| `AffineQuantize(float→int8)` | optimized_ops.h | 6663 | `vfmul` + `vfcvt_x` + clamp |
| `AffineQuantize(float→uint8)` | optimized_ops.h | 6720 | `vfmul` + `vfcvt_x` + clamp |
| `AffineQuantize(float→int16)` | optimized_ops.h | 6778 | `vfmul` + `vfcvt_x` + clamp |
| `ArgMinVector(float)` | optimized_ops.h | 7700 | `vfmin` + index tracking |
| `ArgMinVector(int8)` | optimized_ops.h | 7756 | `vmin` + index tracking |
| `ArgMaxVector(float)` | optimized_ops.h | 7812 | `vfmax` + index tracking |
| `ArgMaxVector(int8)` | optimized_ops.h | 7861 | `vmax` + index tracking |

## Tier 3 — Pooling, Reduction, Resize

| Function | File | Line | RVV Mapping |
|---|---|---|---|
| `AveragePool(uint8)` output division | optimized_ops.h | 3217 | division + `vnclip` |
| `AveragePool(int8)` output division | integer_ops/pooling.h | 288 | division + `vnclip` |
| `MeanImpl` init | reduce.h | 61 | `vmv_v_x` broadcast |
| `MeanImpl` accum loop | reduce.h | 69 | `vsext` + `vadd` + `vredsum` |
| `ResizeBilinear888Uint8` | resize_bilinear.h | 283 | uint8 interpolation (complex) |
| `ResizeBilinearKernel2x2` | resize_bilinear.h | 1356 | `vfmacc` for 4-corner interp |

## Tier 4 — Depthwise Conv Specializations (NEON-geometry-dependent)

These are template specializations for fixed `(input_depth, depth_multiplier)` pairs, using NEON's 128-bit register geometry. The generic accumulation function already has RVV support; these specializations provide marginal additional speedup for specific shapes.

| Function | File | Blocks | Description |
|---|---|---|---|
| `QuantizedDepthwiseConvKernel<*,8,2>` etc. | depthwiseconv_uint8.h | 5 | 15 uint8 kernel specializations |
| `DepthwiseConvInitAccBuffer(uint8)` | depthwiseconv_uint8.h | 1 | bias init for uint8 |
| `DepthwiseConvGeneral` dispatch | depthwiseconv_uint8.h | 3 | kernel selection |
| `FloatDepthwiseConvKernel<*,8,1>` etc. | depthwiseconv_float.h | 2 | 12 float kernel specializations |
| `DepthwiseConvImpl` dispatch | depthwiseconv_float.h | 1 | kernel selection |
| `QuantizedDepthwiseConvKernel(int8)` | integer_ops/depthwise_conv.h | 2 | 15 int8 kernel specializations |
| `DepthwiseConvGeneral` dispatch | integer_ops/depthwise_conv.h | 1 | kernel selection |
| `DepthwiseConvHybrid` | integer_ops/depthwise_conv_hybrid.h | 2 | hybrid quantized conv |
| Transitional depthwise | depthwiseconv_uint8_transitional.h | 4 | transitional implementation |
| 3x3 filter common | depthwiseconv_3x3_filter_common.h | 2 | 3x3 filter utilities |
| 3x3 filter uint8 | depthwiseconv_uint8_3x3_filter.h | 1 | 3x3 filter specialization |

## Tier 5 — Legacy / Rarely Used

| Function | File | Blocks | Description |
|---|---|---|---|
| `FullyConnected` (3 variants) | legacy_optimized_ops.h | 3 | Deprecated; new code uses non-legacy path |
| `ShuffledFullyConnected` (2 variants) | legacy_optimized_ops.h | 2 | Deprecated XOR-shuffle FC |
| `Conv` | legacy_optimized_ops.h | 1 | Deprecated convolution |
| `Softmax` (4 variants) | legacy_optimized_ops.h | 4 | Deprecated softmax |
| `Logistic` | legacy_optimized_ops.h | 1 | Deprecated sigmoid |
| `Tanh` | legacy_optimized_ops.h | 1 | Deprecated tanh |
| `GemmlowpOutputPipelineInt8` | legacy_optimized_ops.h | 1 | Deprecated GEMM pipeline |
| `ShuffledFullyConnected` | optimized_ops.h | 3 | XOR-based shuffled FC |
| `BroadcastPow4D` | optimized_ops.h | 1 | Power operation broadcast |
| `HardSwish(quantized)` | optimized_ops.h | 1 | Quantized HardSwish (complex LUT) |

## Tensor Utilities (in `neon_tensor_utils.cc`)

These are function-level blocks (entire file gated by `#ifdef USE_NEON`), not inline blocks. The RVV equivalents are in `rvv_tensor_utils.cc`. Functions still delegating to Portable:

| Function | RVV Status | Reason |
|---|---|---|
| `ApplySigmoid` | Portable fallback | Needs `gemmlowp::FixedPoint<int16x8_t>` LUT |
| `ApplyTanh` | Portable fallback | Same gemmlowp dependency |
| `MatrixBatchVectorMultiplyAccumulate(int8, asymmetric)` | Portable fallback | Needs row_sums/compute_row_sums logic |
| `MatrixBatchVectorMultiplyAccumulate(int8→int8)` | Portable fallback | Uses CpuBackendGemm |
| `MatrixBatchVectorMultiplyAccumulate(int8→int16)` | Portable fallback | Uses CpuBackendGemm |

## Summary

| Tier | Blocks | Effort | Impact |
|---|---|---|---|
| 1 — Simple element-wise | 12 | Low | High (common ops) |
| 2 — Quantize/Dequantize | 19 | Medium | High (every quantized model) |
| 3 — Pooling/Reduction/Resize | 6 | Medium | Medium |
| 4 — Depthwise conv specializations | 24 | High | Low (generic path has RVV) |
| 5 — Legacy/rarely used | 17 | Medium | Low (deprecated code) |
| **Total** | **78** | | |
