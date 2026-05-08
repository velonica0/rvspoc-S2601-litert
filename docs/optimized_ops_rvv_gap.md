# optimized_ops.h RVV Gap Status

## Current Snapshot

截至 2026-05-07，这个文件里的宏计数更新为：

- `USE_NEON = 71`
- `USE_RVV = 50`

这里要特别注意：

- 这个差值 **不等于** “还差 21 个独立算子”
- 原因是 `optimized_ops.h` 里有很多 `USE_NEON` 是：
  - 同一个函数里的多段 NEON 子块
  - 仅供 NEON 路径调用的 helper
  - AArch64 / LUT / quantized 专用辅助代码

所以后续应继续按“功能块”收敛，而不是只盯计数做机械对齐。

## Goal

目标口径保持不变：

- 在 `tflite/kernels/internal/optimized/optimized_ops.h` 中，
  凡是存在 `USE_NEON` 的热点实现，尽量补一份对应的 `USE_RVV` 路径
- 对量化路径，优先保证 rounding / saturation / clamp 语义一致
- 每一轮都要留下明确的“已补齐 / 未补齐 / 为什么未补齐”的状态

## Added So Far

当前已经补进 `USE_RVV` 的主要块包括：

- `AddElementwise(float)`
- `AddScalarBroadcast(float)`
- `MulElementwise(float)`
- `MulSimpleBroadcast(float)`
- `Div(float)`
- `SubWithActivation(float)`
- `HardSwish(float)`
- `MaximumElementwise(int8_t)`
- `MaximumScalarBroadcast(int8_t)`
- `MinimumElementwise(int8_t)`
- `MinimumScalarBroadcast(int8_t)`
- `PReluScalarBroadcast(float)`
- `PReluElementWise(float)`
- `AddElementwise(uint8_t)`
- `AddScalarBroadcast(uint8_t)`
- `MulElementwise(uint8_t)`
- `MulSimpleBroadcast(uint8_t)`
- `MulElementwise(int32_t)`
- `Quantize(int32 multiplier, int32 shift, ..., uint8_t* output)`
- `Requantize<int8_t, uint8_t>`
- `Requantize<uint8_t, int8_t>`
- `Requantize<int8_t, int8_t>`
- `Requantize<uint8_t, uint8_t>`
- `Dequantize(uint8_t)`
- `Dequantize(int8_t)`
- `Dequantize(int16_t)`
- `ShuffledFullyConnectedWorkerImpl` 的 `batches == 1`
- `ShuffledFullyConnectedWorkerImpl` 的 `batches == 4`
- `ShuffledFullyConnected(...)` 内部 input shuffle
- `AveragePool(uint8_t)` 的 depth inner loop / output clamp-store
- `MaxPool(uint8_t)` 的 depth inner loop / output clamp-store
- `AffineQuantize(int8_t)`
- `AffineQuantize(uint8_t)`
- `AffineQuantize(int16_t)`
- `ArgMinVector(float)`
- `ArgMaxVector(float)`
- `ArgMaxVector(int8_t)`
- `ArgMaxVector(uint8_t)`
- per-channel `Quantize(..., int8_t*)` single-rounding
- per-channel `Quantize(..., int16_t*)` single-rounding
- per-channel `Quantize(..., int8_t*)` double-rounding
- per-channel `Quantize(..., int16_t*)` double-rounding

## Validation

这一轮新增的 RVV 路径已经在 K3 上完成 `optimized_ops.h` 最小 probe 的
`-fsyntax-only` 校验，通过。

本轮实际做过语法校验后的新增项包括：

- `ShuffledFullyConnected` RVV 路径
- `AveragePool(uint8_t)`
- `MaxPool(uint8_t)`
- `AffineQuantize(int8_t/uint8_t/int16_t)`
- `ArgMin/ArgMax`
- per-channel `Quantize(...)`

说明当前主要风险已经从 “intrinsic / 类型是否能编译” 转向：

- 语义一致性
- benchmark 收益
- 个别 helper 的结构性重写

## Remaining High-value Gaps

### 1. Helper-only NEON blocks

这些块解释了计数为什么还没完全追平，但它们不等价于“还差一个完整 kernel”：

- `StoreValue(...)`
- `SaturateAndStore(...)`
- `ScaleWithNewZeroPoint(...)`

其中：

- `StoreValue(...)` 绑定的是 AArch64 LUT softmax 内部的数据布局
- `SaturateAndStore(...)` 绑定的是 quantized HardSwish 的 NEON 路径
- `ScaleWithNewZeroPoint(...)` 只是 `Dequantize` 的 NEON helper

### 2. Quantized HardSwish

严格来说这块更大，因为它不只是 helper，还包括整段 NEON-specialized 算法：

- `HardSwish(const HardSwishParams&, ..., T* output_data)` 的 quantized 分支

这部分虽然很多地方写成 `__ARM_NEON`，不是全部都出现在 `USE_NEON` 计数里，
但从“NEON 做了、RVV 也要有”的口径看，仍然是后续重点。

### 3. Softmax / LUT support

这块剩余工作不止是 helper，还涉及 AArch64-only LUT 流程：

- `FindMaxValue(...)`
- `StoreValue(...)`
- `SoftmaxInt8LUT(...)` 周边 vector helper

这部分要特别小心精度、查表和数据打包语义，不适合只为了追计数硬补。

## Recommended Next Order

后续建议顺序：

1. Quantized HardSwish
2. Softmax / LUT support
3. `ScaleWithNewZeroPoint` 这类零散 helper 收尾
4. 视情况再看 `legacy_optimized_ops.h`、`depthwiseconv_uint8.h` 等文件

## Notes

- 这份文档只跟踪 `optimized_ops.h`，不代表整个 `tflite/` 的 RVV 覆盖率
- 当前很多新增 RVV 路径先保证了“可编译 + 语义收敛”，未必已经是最终性能形态
- 对 per-channel quantize 这类路径，当前 RVV 版本更像“覆盖优先”的过渡实现，
  后面仍可继续把逐 lane 标量计算替换成更实的向量化
