# RVV Optimization Path

## Purpose

这份文档记录当前 LiteRT/TFLite 在 RISC-V RVV 上的优化主线，方便后续继续沿同一条路径推进，而不是每次重新摸索。

文档关注三件事：

- 现在为什么这样做
- 已经优化到了哪里
- 下一步最值得继续补哪些 helper / kernel

如果要专门追踪 `tflite/kernels/internal/optimized/optimized_ops.h` 里
“`USE_NEON` 是否已经有对应 `USE_RVV`” 的收敛状态，单独看
`docs/optimized_ops_rvv_gap.md`。

## Guiding Principles

当前这条路线不是“给所有 kernel 都重写一份 RVV 专核”，而是按收益和改动规模分层推进：

1. 先保证所有 CPU kernel 在 RISC-V 上可构建、可运行、可 fallback。
2. 优先优化共享 helper，因为一处改动会影响多个上层算子。
3. 接口组织尽量贴近现有 NEON/SSE 处理方式，避免把 RVV 分发逻辑堆进 `reference` 路径。
4. 每新增一条 RVV 路径，都要有 scalar 对照验证和 benchmark 数据。

## Current Dispatch Design

当前推荐保留的接入方式是“小改动、仿照 NEON/SSE”：

- `tflite/kernels/internal/tensor_utils.cc`
  - 负责按平台选择 `sse` / `neon` / `rvv` / `reference`
- `tflite/kernels/internal/optimized/rvv_tensor_utils.h`
  - 对外提供 RVV wrapper，使用 `RVV_OR_PORTABLE(...)`
- `tflite/kernels/internal/optimized/rvv_tensor_utils.cc`
  - 放 RVV 实现
- `tflite/kernels/internal/reference/portable_tensor_utils.h`
  - 保持纯 portable/reference，不再塞 RVV 分发逻辑

这条路径的好处是：

- 和现有 NEON 结构一致，后续维护成本更低
- 回退逻辑清晰，不容易污染 reference 路径
- 更容易逐个 helper 扩展覆盖面

## Implemented So Far

当前已经落地并有 benchmark 的 RVV helper 包括：

- float `VectorVectorDotProduct`
- float `ReductionSumVector`
- int8 `ReductionSumVector`
- `SymmetricQuantizeFloats`
- `AsymmetricQuantizeFloats`
- int8 `MatrixScalarMultiplyAccumulate`
- float `MatrixBatchVectorMultiplyAccumulate`
- int8 `MatrixBatchVectorMultiplyAccumulate`
- int8 `MatrixBatchVectorMultiplyAccumulate` 的 `per_channel_scale + input_offset` 路径
- int8 `MatrixBatchVectorMultiplyAccumulate(... -> int16_t)`
- int8 `MatrixBatchVectorMultiplyAccumulate(... -> int8_t)`
- float `SparseMatrixBatchVectorMultiplyAccumulate1x4`
- float ledger `SparseMatrixBatchVectorMultiplyAccumulate`
- int8 `SparseMatrixBatchVectorMultiplyAccumulate1x16`
- int8-to-float ledger `SparseMatrixBatchVectorMultiplyAccumulate`
- int16 `ApplyLayerNorm`
- int16 `ApplySigmoid`
- int16 `ApplyTanh`
- int16 `CwiseMul`
- int8 output `CwiseMul`
- int16 `CwiseAdd`
- `CwiseClipping<float/int16/int8>`
- int16 `VectorBatchVectorCwiseProductAccumulate`
- `Sub1Vector<float/int16>`
- `VectorScalarMultiply<int8 -> float>`
- `MeanStddevNormalization<float>`
- `IsZeroVector<float/int8>`

这些 helper 已经直接影响到多类上层算子：

- `fully_connected`
- `conv`
- `batch_matmul`
- `svdf`
- hybrid recurrent / LSTM 的一部分公共路径

最新 benchmark 结果见 `docs/rvv_scalar_benchmarks.md`。

当前各个 public helper 的 “NEON 对齐程度 / VLEN 处理方式 / 已实测状态” 总表见
`docs/rvv_tensor_utils_status.md`。

## Structural Validation Baseline

后续继续补 RVV 时，至少保留下面两类结构验证：

1. `tensor_utils.cc` 的 RVV 语法检查
2. public helper dispatch probe 的编译 / 链接 / 运行验证

原因是 helper benchmark 直接编译 `optimized/rvv_tensor_utils.cc`，只能说明 RVV 数学实现本身可用；它不能自动证明 `tensor_utils.cc` 的公共分发已经接好。

## Current Gaps Against NEON

和 `tflite/kernels/internal/optimized/neon_tensor_utils.h` 对比，当前 RVV 在 `rvv_tensor_utils.h` 这一层的 public helper 覆盖已经基本对齐 NEON；主要差距已经不再是“有没有 helper 入口”，而是“哪些 helper 还可以做得更深、更快”。

### Priority A: Recurrent / LSTM Helper Parity

这组最值得优先补，因为调用面集中、收益高，而且很符合“继续沿 tensor_utils 这条线扩覆盖”的目标。

这一组里，当前已经接入并跑过 benchmark 的有：

- `ApplyLayerNorm`
- `ApplySigmoid`
- `ApplyTanh`
- `CwiseMul`
- `CwiseAdd`
- `CwiseClipping`
- `VectorBatchVectorCwiseProductAccumulate`
- `MeanStddevNormalization`
- `Sub1Vector`
- `VectorScalarMultiply`
- `IsZeroVector`

这一组接下来最值得继续补的，不再是“从 0 到 1 接入”，而是：

- `ApplyLayerNorm`
  这一轮已经把 second pass 的精确量化后处理补成了 RVV 路径，K3 上速度从接近持平提升到约 `2.47x ~ 2.84x`。后续如果还要继续逼近 NEON，更值得看的将是 cache/locality、bias/weight 装载，以及是否能继续减少 64-bit 中间态开销。
- `ApplySigmoid`
  当前已经改成 bit-exact LUT 快速路径，K3 上约 `54.27x ~ 104.65x`。这条路已经很快，但严格来说它仍不是“NEON 同构”的 RVV fixed-point 向量实现；如果后面一定要追求结构完全看齐，需要补 `gemmlowp` 的 RVV vector type 支撑。
- `ApplyTanh`
  状态和 `ApplySigmoid` 类似，当前通过 exact LUT 路径达到约 `58.79x ~ 96.48x`，性能已经很好，但结构上仍不是 NEON 那种 `FixedPoint<vector>` 实现。

这些 helper 主要影响：

- `tflite/kernels/lstm_eval.cc`
- `tflite/kernels/lstm.cc`
- `tflite/kernels/unidirectional_sequence_lstm.cc`

如果要最大化“尽可能多算子”的体感收益，这一组通常比再去单独扩一个小 kernel 更划算。

### Priority B: Hybrid Quantization / Precheck

这组 helper 不一定是单次算得最重的，但它们处在很多 hybrid kernel 的前处理路径上，覆盖面非常大。

这一组里，当前已经补齐并有 benchmark / dispatch 验证的有：

- `SymmetricQuantizeFloats`
- `AsymmetricQuantizeFloats`
- `IsZeroVector`

主要影响：

- hybrid `fully_connected`
- hybrid `conv`
- hybrid `depthwise_conv`
- `batch_matmul`
- `transpose_conv`

因此这一层和 NEON 的“共享量化 helper 入口是否存在”这一差距已经补平；后续如果继续沿 NEON 看齐，重点应转向更深的 recurrent 向量化和算子级热点。

### Priority C: Sparse Shared Helper

这一组在 wrapper 覆盖上已经补齐，并且已有 benchmark / dispatch 验证：

- `SparseMatrixBatchVectorMultiplyAccumulate1x4`
- `SparseMatrixBatchVectorMultiplyAccumulate`
- `SparseMatrixBatchVectorMultiplyAccumulate1x16`

主要影响：

- sparse `fully_connected`
- sparse recurrent / LSTM

接下来这一组的重点不再是“从 0 到 1 接入”，而是：

- float sparse 的 block 累加是否进一步减少小 block 开销
- int8 sparse 的 dot + row-sum / quant 后处理是否继续做更深的向量化
  这一轮已经把 `1x16` 的 row sum 预计算移出 batch 循环，并在两条 int8 sparse 路径里加了双累加器；K3 上 `SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>` 已到约 `2.10x ~ 3.36x`，`SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>` 到约 `1.54x ~ 2.12x`，但相对 dense path 仍有继续深挖空间。
- 针对更稀疏或更大的实际模型 shape，继续找最有效的 block/layout 路径

### Priority D: Int8 LSTM Gate / Projection Helper

这一组里，当前已经补齐并有 benchmark / dispatch 验证的有：

- `MatrixBatchVectorMultiplyAccumulate(... int16_t* output ...)`
- `MatrixBatchVectorMultiplyAccumulate(... int8_t* output ...)`

仍然还可以继续看的有：

- `MatrixBatchVectorMultiply(...)`
- `TwoGateSaturatingAdd`

这组更偏量化 LSTM / sequence LSTM。

## Recommended Execution Order

后续建议按下面顺序推进。

### Stage 1: Finish Shared Helper Parity

目标：

- 保持 RVV `tensor_utils` 与 NEON 的 public helper 覆盖对齐
- 继续保持“小改动 + 公共 helper 扩散到多算子”的路线

优先顺序：

1. recurrent / LSTM helper
2. sparse matvec helper 的更深向量化
3. int8 gate / projection helper 的更深后处理向量化
4. hybrid quantization / zero-check 的零散边角

### Stage 2: Move To Operator Families

在 helper 路线吃到第一波大收益后，再做算子级 RVV。

推荐顺序：

1. `fully_connected`
2. `svdf`
3. `unidirectional_sequence_lstm` / recurrent 系列
4. `batch_matmul`
5. `depthwise_conv`
6. `softmax` / `log_softmax`
7. `pooling`
8. 高频 elementwise：`add` / `mul` / `sub`
9. `reduce/mean`
10. `resize_bilinear` / `resize_nearest_neighbor`
11. 低比特 `4bit fully_connected`

## First Operator-Level Step

当前更推荐先从 `fully_connected` 家族切入，而不是一上来就做
`depthwise_conv`。

原因是：

- 现有 `tensor_utils` RVV helper 已经覆盖 dense matvec、hybrid quant、
  sparse matvec、row sum、activation 等关键积木。
- `fully_connected` / `svdf` / `lstm_eval` 这些上层算子对
  `tensor_utils` 依赖非常直接，能最快把 helper 级收益变成算子级收益。
- 相比 `conv/depthwise_conv`，这批算子更少依赖额外的 packing /
  blocking / im2col 专核设计，更适合先落第一批 operator-level RVV。

目前已经开始的第一步就是：

- 在 RVV 目标上，把普通 float dense `FullyConnected` 的 generic optimized
  路径显式切到 `tensor_utils` 驱动的 `EvalPie` 路径，让 top-level
  `fully_connected` 直接吃到 RVV `MatrixBatchVectorMultiplyAccumulate`
  和 activation helper 的收益。
- 在 RVV 目标上，把 `tflite/kernels/lstm_eval.cc` 里本地 float gate /
  projection matvec helper 也显式切到
  `tensor_utils::MatrixBatchVectorMultiplyAccumulate`，让
  `unidirectional_sequence_lstm` / recurrent float 路径直接复用同一套 RVV
  matvec 主干。

等这批 helper-driven 算子稳定以后，再进入 `depthwise_conv` 会更顺。

## Current Operator-Level Status

以 `2026-05-07` 这轮状态为准，算子级 RVV 已经落地到下面几块：

- `fully_connected`
  普通 float dense generic optimized 路径在 RVV 目标上显式转向
  `EvalPie` / `tensor_utils` 栈。
- `lstm_eval`
  本地 float gate/projection matvec helper 在 RVV 目标上不再走
  `optimized_ops::FullyConnected`，而是直接走
  `tensor_utils::MatrixBatchVectorMultiplyAccumulate`。
- `optimized_ops` 高频 float elementwise
  非 broadcast `Add` / `Mul` / `SubWithActivation` / `Div` 已补上 RVV
  主循环；K3 上新增 operator benchmark 显示：
  - `Add<float>`: `6.09x ~ 8.82x`
  - `Mul<float>`: `4.01x ~ 5.35x`
  - `SubWithActivation<float>`: `6.08x ~ 8.76x`
  - `Div<float>`: `2.00x ~ 2.01x`

当前还需要单独跟进的一点是：

- `BroadcastAddDispatch<float>` / `BroadcastMulDispatch<float>` 的 1D
  scalar-broadcast case 在 K3 上目前仍接近 `1.00x`，说明虽然后面的
  scalar-broadcast inner kernel 已补 RVV，但 top-level broadcast dispatch
  是否稳定落到那条快路径，或者 wrapper 开销是否掩盖收益，还需要继续拆。

## File Map For Future Work

后续继续推进时，最常用到的文件入口如下：

### Dispatch / Wrapper

- `tflite/kernels/internal/tensor_utils.cc`
- `tflite/kernels/internal/optimized/rvv_tensor_utils.h`
- `tflite/kernels/internal/optimized/rvv_check.h`

### RVV Implementation

- `tflite/kernels/internal/optimized/rvv_tensor_utils.cc`
- `tflite/kernels/internal/optimized/rvv_tensor_utils_impl.h`

### Portable / Reference Baseline

- `tflite/kernels/internal/portable_tensor_utils.h`
- `tflite/kernels/internal/reference/portable_tensor_utils.h`
- `tflite/kernels/internal/reference/portable_tensor_utils_impl.h`

### High-Value Call Sites

- `tflite/kernels/lstm_eval.cc`
- `tflite/kernels/fully_connected.cc`
- `tflite/kernels/conv.cc`
- `tflite/kernels/depthwise_conv.cc`
- `tflite/kernels/internal/kernel_utils.cc`
- `tflite/kernels/activations.cc`

## Benchmark / Validation Workflow

建议后续沿用下面的闭环：

1. 先补 helper 或 kernel 的 RVV 实现
2. 过 `tensor_utils.cc` 或对应入口的语法检查
3. 过最小 dispatch probe 或最小单测
4. 增加 scalar vs RVV benchmark
5. 把结果记到文档里

当前已有 benchmark 脚本：

- `tools/riscv/run_rvv_tensor_utils_benchmark.sh`
- `tools/riscv/run_rvv_operator_benchmark.sh`

当前已有结果文档：

- `docs/rvv_scalar_benchmarks.md`
- `docs/rvv_operator_benchmarks.md`
- `docs/findings.md`
- `docs/decisions.md`
- `docs/gotchas.md`

## Practical Definition Of Done

本项目里“优化路径完成”更现实的定义是：

- 所有现有 CPU kernel 在 RISC-V 上可运行、可 fallback
- 共享 helper 这条线覆盖到主要热点家族
- 重点模型真正会跑到的 kernel 家族都有 RVV 路径
- 每条新增 RVV 路径都有可复现的 scalar 对照结果

## Short Version

如果后续只记一句话，可以记成：

先沿 `tensor_utils` 继续补齐和 NEON 最接近的 helper，优先把
`fully_connected / svdf / lstm` 这批 helper-driven 算子正式挂到 RVV
路径上；现在已经开始进入 `optimized_ops` 的 float elementwise
`add/mul/sub/div`，后续再继续往 `depthwise_conv -> softmax/log_softmax ->
pooling -> broadcast/更复杂 elementwise` 这条更重的算子级 RVV 路线推进。
