# RISC-V Kernel Coverage Plan

详细的 RVV 推进主线、已覆盖 helper、下一批优先项见 `docs/rvv_optimization_path.md`。

## Reality Check

LiteRT/TFLite 这里不是只有少数几个 kernel。

- `tflite/kernels/*.cc`: 319 个源文件
- `tflite/kernels/internal/**/*.cc`: 54 个源文件

如果目标理解成“给全部 kernel 都各写一份 RVV 专用实现”，这在比赛周期内并不现实，也不是收益最高的路线。

更合理的目标应该拆成两层：

1. 所有 CPU kernel 在 RISC-V 上都能编译、运行、回退到 scalar/reference 路径。
2. 所有热点 kernel 家族都有 RVV 优化路径，覆盖比赛模型真正会跑到的算子。

## Current Status

当前已经完成的是：

- RISC-V / RVV 构建入口
- float `VectorVectorDotProduct`
- float `ReductionSumVector`
- int8 `ReductionSumVector`
- int8 `MatrixScalarMultiplyAccumulate`
- float / int8 `MatrixBatchVectorMultiplyAccumulate`
- int8 `MatrixBatchVectorMultiplyAccumulate` 的 `per_channel_scale + input_offset` 路径
- RVV vs scalar 的多 helper 可运行 benchmark

当前还没有完成的是：

- sparse / low-bit / activation / cwise 等更多共享 tensor utils RVV 路径
- `depthwise_conv`、pooling、softmax、elementwise 等更多算子家族的成组覆盖
- “全部已有 kernel 测试” 在 K3 上的系统化跑通

## Coverage Strategy

### Layer 1: Full Runtime Coverage

目标：不是所有 kernel 都加速，而是所有 kernel 都能在 RISC-V 上跑通。

手段：

- 保持 portable/reference 路径可用
- 补齐 RISC-V 下缺失的 build flags、include path、test target
- 逐步跑通已有 kernel tests，先从 internal helpers 和高频 builtin ops 开始

### Layer 2: Shared Helper Coverage

优先补共享 helper，因为一处优化会影响多个上层算子：

- float/int8 `MatrixBatchVectorMultiplyAccumulate`
- `VectorVectorDotProduct`
- `ReductionSumVector`
- `MatrixScalarMultiplyAccumulate`
- activation / clipping / cwise helper

### Layer 3: Hot Operator Family Coverage

这些算子家族最值得优先做 RVV：

- `fully_connected`
- `conv`
- `depthwise_conv`
- `batch_matmul`
- `pooling`
- `softmax`
- 高频 elementwise：`add` / `mul` / `sub` / `div` / `relu`

### Layer 4: Long Tail Coverage

这些应该最后处理：

- 稀疏路径
- 4bit / low-bit 特化路径
- 控制流、hashtable、随机数、训练相关冷门算子
- 只在极少数模型触发的 custom / variant kernels

## Next Execution Order

1. 跑通 K3 上可重复执行的 shared-helper benchmark/test。
2. 把 int8 `tensor_utils` 路径补齐并记录 RVV/scalar 数据。
3. 以 `fully_connected` / `conv` / `depthwise_conv` / `batch_matmul` 为第一批家族做算子级 benchmark。
4. 开始系统化跑 builtin op tests，记录哪些 kernel 已“可运行”、哪些已“已加速”。

## Definition of Done

“所有 kernel” 对本项目更可落地的完成定义应为：

- 所有现有 CPU kernel 在 RISC-V 上可构建、可运行、可 fallback。
- 比赛模型涉及的所有热点 kernel 都有 RVV 加速。
- 每个已加速 kernel 都有 scalar 对照测试和 benchmark 数据。
