# Unoptimized Operators (USE_NEON blocks without RVV)

**63 RVV blocks** implemented across the codebase. Tier 1 fully done, Tier 2 fully done, Tier 3 mostly done.

## Tier 1 - Simple Element-wise: DONE (8/8)

MaximumElementwise, MaximumScalarBroadcast, MinimumElementwise, MinimumScalarBroadcast, PReluElementWise, PReluScalarBroadcast, AddScalarBroadcast(float), MulSimpleBroadcast(float).

## Tier 2 - Quantization / Dequantization: DONE (12/12)

- Dequantize: uint8->float, int8->float, int16->float
- AffineQuantize: float->int8, float->uint8, float->int16
- Requantize: int8->uint8, uint8->int8, int8->int8, uint8->uint8

**Remaining** (not worth vectorizing or too specialized):

| Function | File | Reason |
|---|---|---|
| AddScalarBroadcast(uint8) | optimized_ops.h | Quantized broadcast - same pattern as existing uint8 Add |
| MulElementwise(uint8) | optimized_ops.h | Quantized uint8 mul - same pattern as existing uint8 Add |
| MulSimpleBroadcast(uint8) | optimized_ops.h | Quantized broadcast - same pattern |
| FindMaxValue | optimized_ops.h | Inside TFLITE_SOFTMAX_USE_UINT16_LUT guard (aarch64+clang only) |
| ArgMinVector/ArgMaxVector (4) | optimized_ops.h | Requires index tracking with reduction - complex |

## Tier 3 - Pooling, Reduction, Resize: DONE (5/6)

MeanImpl(int8), MeanImpl(uint8), ResizeBilinearKernel(float), ResizeBilinearKernel2x2(float).

**Remaining**:

| Function | File | Reason |
|---|---|---|
| ResizeBilinear888Uint8 | resize_bilinear.h | ~1000 lines of complex uint8 8x8 interpolation with NEON-specific struct types |

## Tier 4 - Depthwise Conv Specializations: SKIPPED

24 NEON template specializations for fixed (input_depth, depth_multiplier) pairs. The generic accumulation function already has RVV support. These specializations are deeply tied to NEON's 128-bit register geometry and provide marginal additional speedup for specific shapes only.

## Tier 5 - Legacy / Rarely Used: SKIPPED

17 blocks in legacy_optimized_ops.h (deprecated FullyConnected, Conv, Softmax, Logistic, Tanh) and 5 blocks in optimized_ops.h (ShuffledFullyConnected, BroadcastPow4D, quantized HardSwish).

## Tensor Utilities: 5 functions on Portable fallback

| Function | Reason |
|---|---|
| ApplySigmoid | Needs gemmlowp::FixedPoint<int16x8_t> LUT |
| ApplyTanh | Same gemmlowp dependency |
| MatrixBatchVectorMultiplyAccumulate(int8, asymmetric) | Needs row_sums logic |
| MatrixBatchVectorMultiplyAccumulate(int8->int8) | Uses CpuBackendGemm |
| MatrixBatchVectorMultiplyAccumulate(int8->int16) | Uses CpuBackendGemm |
