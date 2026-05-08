# RVV Remaining Gap Status

## Scope

这份文档只回答一个问题：

- 截至 `2026-05-08`，仓库里还有哪些 RVV 缺口没有补完

判断口径分两层：

1. 已经有真实 RVV 路径并且已经做过 scalar 对照 benchmark 的，不再算缺口。
2. 仍然只有 scalar / generic，或者只有 NEON / AArch64 特化而没有 RVV 对应的，算缺口。

这里特别避免把“看起来还在判断 `USE_NEON`，但 generic path 实际已经能吃到 `USE_RVV`”的文件误报成缺口。

## Already Covered

下面这些已经不应再算“未做 RVV”：

- `tflite/kernels/internal/optimized/rvv_tensor_utils.cc`
  - 共享 helper 主线已经有 RVV 实现和 benchmark。
- `tflite/kernels/internal/optimized/optimized_ops.h`
  - 已覆盖的热点包括 float `Add` / `Mul` / `SubWithActivation` / `Div`
  - 已覆盖 affine quantize、dequant-style helper、uint8 pooling、HardSwish、ArgMin/ArgMax
- `tflite/kernels/fully_connected.cc`
  - dense float `EvalPie` 路径已经接到 RVV 侧 benchmark。
- `tflite/kernels/lstm_eval.cc`
  - float gate / output-projection 路径已经接到 RVV 侧 benchmark。
- `tflite/kernels/internal/optimized/integer_ops/*.h`
  - 已覆盖 `add` / `mul` / `sub` / `pooling` / `depthwise_conv` / `leaky_relu` / `lut` / `mean`

当前可直接参考的实测数据：

- `docs/rvv_scalar_benchmarks.md`
- `docs/rvv_operator_benchmarks.md`

## Important Non-Gaps

下面这些文件虽然还保留 `USE_NEON` 注册分支，但当前不能算 RVV 缺口：

- `tflite/kernels/add.cc`
- `tflite/kernels/mul.cc`
- `tflite/kernels/sub.cc`
- `tflite/kernels/div.cc`

原因不是它们“已经都有独立 RVV 顶层注册”，而是：

- `add` / `mul` / `sub` / `div` 在 RISC-V 上走的是 `kGenericOptimized`
- 而对应的 `optimized_ops.h` 主循环已经补了 `USE_RVV`

所以对这几类算子，真正的缺口不在顶层 `Register_*`，而在还没补到 RVV 的更深内部实现上。

## Confirmed High-Value Gaps

### P0: Conv Family Still Has No Source-Level RVV Path

关键文件：

- `tflite/kernels/conv.cc`
- `tflite/kernels/internal/optimized/multithreaded_conv.h`

当前状态：

- 这条家族没有出现 `USE_RVV` 或 `__riscv_vector` 分支。
- 当前 `conv` 只能间接受益于先前补好的 shared `tensor_utils` helper。
- 但 `conv` 自己的 `im2col` / GEMM / multithreaded optimized 路径还没有 RVV 专门实现。

为什么这是高优先级缺口：

- `conv` 是比赛模型和通用 CNN 模型的核心热点。
- 仅靠 helper 级收益，不等于 top-level `conv` 已经完成 RVV 化。

建议后续拆分 benchmark：

- `Conv<float>`
- `Conv<int8 per-channel>`
- `Hybrid Conv`

### P0: Float Depthwise Conv Still Falls Back To Non-RVV Path

关键文件：

- `tflite/kernels/depthwise_conv.cc`
- `tflite/kernels/internal/optimized/depthwiseconv_float.h`

当前状态：

- `depthwise_conv.cc` 在 RISC-V 上会走 `kGenericOptimized`。
- 但 `depthwiseconv_float.h` 的快路径选择仍然只有 `#ifdef USE_NEON` 的 float depthwise kernels。
- 没命中 NEON 快核时，最终回到通用慢路径。

为什么这是高优先级缺口：

- `depthwise_conv` 是 MobileNet / depthwise-heavy 网络的关键热点。
- 当前 int8 per-channel `integer_ops/depthwise_conv.h` 已有 RVV，但 float depthwise 还没有。

建议后续拆分 benchmark：

- `DepthwiseConv<float>` stride1 / stride2
- 不同 `input_depth` / `depth_multiplier` 组合

### P0: Quantized Softmax / LogSoftmax Still Missing RVV

关键文件：

- `tflite/kernels/activations.cc`
- `tflite/kernels/internal/optimized/optimized_ops.h`

当前状态：

- `optimized_ops.h` 中的 `SoftmaxInt8LUT`、`FindMaxValue`、`StoreValue` 仍是 NEON / AArch64 LUT 思路。
- 这部分没有 `USE_RVV` 对应实现。
- 文档 `docs/optimized_ops_rvv_gap.md` 里已经把 softmax / LUT support 标成剩余重点。

为什么这是高优先级缺口：

- softmax / log_softmax 常在分类尾部直接命中。
- LUT、max-reduction、quantized normalize 这几段都容易产生精度和性能双重问题，不能简单当尾巴处理。

建议后续拆分 benchmark：

- `Softmax<int8>`
- `Softmax<uint8>`
- `LogSoftmax<float>`
- `LogSoftmax<int8>`

## Confirmed Medium-Priority Gaps

### P1: Generic Reduce / Mean Family Has Partial RVV Coverage, Generic Reductions Pending

关键文件：

- `tflite/kernels/reduce.cc`
- `tflite/kernels/internal/optimized/reduce.h`

当前状态：

- `reduce.cc` 默认注册的是 `kGenericOptimized`。
- `reduce.h` 里的 `MeanImpl(uint8_t)` 现在已经补了 RVV depth 向量累加路径。
- `Mean<float>` 在“只 reduce 最后一维”的快路径上现在已经补了 RVV reduction。
- 对应算子级 benchmark 已经补齐，结果见：
  - `docs/rvv_operator_benchmarks.md`
  - `docs/rvv_scalar_benchmarks.md`
- 当前实测结果：
  - `MeanImpl<uint8>`：1.26x 到 1.62x，加速同时保持 `max_abs_diff=0`
  - `Mean<float>` last-dim：中大 shape 上 1.82x 到 1.85x，`max_abs_diff<=2.4e-7`
- 通用 `ReduceGeneric` 族本身仍然还没有 RVV 专门实现。
- 这和已经完成的 `integer_ops/mean.h` 不同，后者只覆盖了 int8 height-width reduction 那一支。

剩余缺口：

- `SUM` / `REDUCE_MAX` / `REDUCE_MIN` / `REDUCE_PROD` / `ANY` / `ALL`
  当前还没有形成系统的 RVV 加速。
- `MEAN` 也只是补到了更常见的 uint8 与 float last-dim 热路径，还不是完整覆盖。

后续仍建议补的 benchmark：

- `ReduceSum<float>`
- `ReduceMax<int8>`

### P1: Resize Bilinear Has Partial RVV Coverage, Broader Kernels Pending

关键文件：

- `tflite/kernels/internal/optimized/resize_bilinear.h`

当前状态：

- 文件里原本主要是 `USE_NEON` 专用 kernel。
- `ResizeBilinear(float)` 的 generic `ResizeBilinearKernel` 现在已经补了 RVV depth 向量乘加。
- `ResizeBilinear(uint8)` 的 `ResizeBilinearGenericSmallChannel<uint8_t>` 现在已经补了 RVV depth 向量插值路径。
- 对应算子级 benchmark 已经补齐，结果见：
  - `docs/rvv_operator_benchmarks.md`
  - `docs/rvv_scalar_benchmarks.md`
- 当前实测结果：
  - `ResizeBilinear<uint8>` generic-small-channel：2.04x 到 2.20x，`max_abs_diff=0`
  - `ResizeBilinear<float>` generic operator path：结果数值完全一致，但整体算子级 speedup 约 1.00x，说明当前瓶颈更多还在坐标/插值调度外围而不是 depth 累加内核
- `ResizeBilinear(int8)` 旁边仍然留着明确 TODO：
  - `Create optimized int8 version from uint8`

剩余缺口：

- resize 在检测、分割、图像预后处理链里很常见。
- bilinear 的访存和插值结构比较适合做 RVV lane 级展开。
- 但目前还没有覆盖 2x2 special float path、legacy uint8 scale-8 特化之外的更多专门 kernel，也没有 int8 RVV 路径。

后续仍建议补的 benchmark：

- `ResizeBilinear<float>` 2x2 upsample
- `ResizeBilinear<uint8>` scale=8 fast path
- `ResizeBilinear<int8>` generic path

### P1: ResizeNearestNeighbor Has Partial RVV Coverage, Benchmarks Pending

关键文件：

- `tflite/kernels/resize_nearest_neighbor.cc`
- `tflite/kernels/internal/optimized/optimized_ops.h`

当前状态：

- 顶层 `Register_RESIZE_NEAREST_NEIGHBOR()` 在 RISC-V 上会选 `kGenericOptimized`。
- `optimized_ops::ResizeNearestNeighbor(uint8)` 的 depth copy 内环现在已经补了 `USE_RVV` 向量拷贝。
- 但 `align_corners=true` / `half_pixel_centers=true` 仍然会走 reference path。

剩余缺口：

- 目前还缺按算子拆分的 scalar vs RVV benchmark。
- 更复杂的坐标分支还没有继续扩到 RVV。

建议后续拆分 benchmark：

- `ResizeNearestNeighbor<uint8>`
- 不同 `align_corners` / `half_pixel_centers` 组合

### P1: Dequantize Wiring Is Fixed, Benchmarks Pending

关键文件：

- `tflite/kernels/dequantize.cc`
- `tflite/kernels/internal/optimized/optimized_ops.h`

当前状态：

- `optimized_ops.h` 里的 `Dequantize(uint8_t/int8_t/int16_t -> float)` 已经补了 `USE_RVV`。
- `dequantize.cc` 的 `Register_DEQUANTIZE()` 现在已经把 `USE_RVV` 也导向 optimized。
- 这项已经不再是“实现不可达”的 wiring gap。

剩余缺口：

- 还缺单独的 scalar vs RVV benchmark。
- benchmark 完成前，这一项还不能按本文口径完全移出 gap 清单。

建议后续动作：

1. 补 `Dequantize<uint8/int8/int16 -> float>` benchmark
2. benchmark 完成后把这一项移出 P1

### P1: Cast Has Partial RVV Coverage, Benchmarks Pending

关键文件：

- `tflite/kernels/cast.cc`

当前状态：

- `copyCast(const float* -> int32_t/int16_t/uint8_t)` 现在已经补了 RVV 路径。
- 其它更长尾的 cast 族仍然主要走标量实现。

剩余缺口：

- 还没有 operator-level benchmark。
- `float -> int8` / 其它类型组合还没有专门的 RVV 快路径。

## Confirmed Long-Tail / Specialized Gaps

### P2: Legacy Uint8 Depthwise 3x3 / Transitional Kernels

关键文件：

- `tflite/kernels/internal/optimized/depthwiseconv_uint8.h`
- `tflite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h`
- `tflite/kernels/internal/optimized/depthwiseconv_uint8_transitional.h`
- `tflite/kernels/internal/optimized/depthwiseconv_3x3_filter_common.h`

当前状态：

- 这里仍然是 AArch64 / NEON 3x3 dot-product 和 transitional kernel 体系。
- 目前完成的 RVV depthwise 是 `integer_ops/depthwise_conv.h` 那一套 int8 per-channel 路径，不等于 legacy uint8 这套也完成了。

为什么放在 P2：

- 旧 uint8 路径现在的重要性低于 int8 per-channel 和 float depthwise。
- 但如果要追“更完整的 kernel 覆盖”，这块迟早还得补。

### P2: 4bit FullyConnected Still ARM/SSE-Only

关键文件：

- `tflite/kernels/internal/optimized/fully_connected_4bit.h`
- `tflite/kernels/internal/optimized/4bit/neon_fully_connected*.{h,cc}`

当前状态：

- 当前 4bit FC 只接了 SSE 或 NEON 后端。
- 没有 RVV 4bit 后端。

为什么放在 P2：

- 这是明显的专项优化，不是当前主线模型最先要吃到的收益。
- 但从“尽可能多 kernel”角度，它是一个完整的空白项。

### P2: legacy_optimized_ops.h Still Contains Large NEON-Only Long Tail

关键文件：

- `tflite/kernels/internal/optimized/legacy_optimized_ops.h`

当前状态：

- 这个文件里仍有大量旧版 NEON 专核。
- 当前主线已经更多转向 `optimized_ops.h`、`integer_ops/*` 和 `tensor_utils`。

为什么放在 P2：

- 这块更像兼容层 / 旧路径收尾。
- 真要追它，必须按调用面再次筛选，不能直接机械平移所有 NEON 分支。

## Suggested Next Order

如果下一轮要继续补 RVV，建议顺序如下：

1. `conv`
2. `softmax / log_softmax`
3. `float depthwise_conv`
4. `dequantize` dispatch wiring
5. `reduce / mean`
6. `resize_bilinear / resize_nearest_neighbor`
7. `cast`
8. `legacy uint8 depthwise`
9. `4bit fully_connected`
10. `legacy_optimized_ops` 长尾收尾

## Short Version

如果只记一句话：

- 已经做完的一批主要是 `tensor_utils`、`optimized_ops` 高频 elementwise、`fully_connected`、`lstm_eval` 和 `integer_ops`；
- 真正还没补完的核心 RVV 缺口，现在主要集中在 `conv`、`softmax/log_softmax`、`float depthwise_conv`、`reduce/mean`、`resize`、`dequantize` wiring，以及 `4bit/legacy` 这批长尾路径。
