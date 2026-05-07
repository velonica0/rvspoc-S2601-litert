# RVV Tensor Utils Status

## Purpose

这张表专门回答两个问题：

- `rvv_tensor_utils` 里的各个 helper，当前到底有多大程度是“按 NEON 路线来的”
- 这些 helper 对 `VLEN=128/256/512` 的支持，是“代码设计支持”还是“已经实测支持”

## Reading Guide

| Label | Meaning |
| --- | --- |
| `Strict NEON-style` | 数学流程或 helper 结构刻意贴近 `neon_tensor_utils` / `gemmlowp`，不是简单追求“结果一样” |
| `Semantic parity` | 行为、接口、量化语义与 NEON/portable 对齐，但 RVV 实现不是 line-by-line 仿写 |
| `RVV-native` | 更偏 RVV 本地化实现，没有必要硬对齐 NEON 结构 |
| `Dynamic vsetvl` | 主循环按运行时 `vsetvl` / `vsetvlmax` 自适应，不写死 lane 数 |
| `Format-fixed block` | 数据布局本身固定 block 大小，例如 4 或 16；代码仍可在不同 VLEN 上运行，但不会因为 VLEN 变宽就自动扩大 block |

## Global VLEN Notes

| Item | Current Status |
| --- | --- |
| Runtime-verified VLEN | 目前完整 benchmark 只在 `VLEN=256` 的 K3 上跑过，见 `docs/rvv_scalar_benchmarks.md` |
| Design target range | 当前 RVV helper 明确按 `VLEN=128/256/512` 目标范围实现 |
| Build contract | 使用 `-march=rv64gcv_zvl128b`，表示最小 VLEN 下限为 128 bit，不是把运行时 VLEN 锁死为 128 |
| Dynamic execution | 大部分 helper 通过 `vsetvl` / `vsetvlmax` 在运行时自适应 `vl` |
| Explicit upper bound | `tflite/kernels/internal/optimized/rvv_tensor_utils.cc` 里有 `kMaxRvvBits = 512`，用于少量临时 buffer 上限，因此当前不承诺 `VLEN > 512` |
| Important caveat | “支持不同 VLEN”不等于“所有 helper 都能随着更宽 VLEN 线性吃满性能”；sparse block 类 helper 受数据布局限制更大 |

## Status Table

| Helper family | Current RVV status | NEON alignment | VLEN handling | `128/256/512` design support | Runtime verified | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `IsZeroVector<float/int8>` | Implemented + benchmarked | `RVV-native` | `Dynamic vsetvl` | Yes | `256 only` | 更偏基础 precheck helper，不需要刻意模仿 NEON |
| `SymmetricQuantizeFloats` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvl` | Yes | `256 only` | 输出、scale 等与 scalar 精确对齐 |
| `AsymmetricQuantizeFloats` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvl` | Yes | `256 only` | 输出、scale、offset 与 scalar 精确对齐 |
| `VectorVectorDotProduct<float>` | Implemented + benchmarked | `RVV-native` | `Dynamic vsetvl` | Yes | `256 only` | 属于共享 dense math helper |
| `ReductionSumVector<float>` | Implemented + benchmarked | `RVV-native` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 浮点误差主要来自累加顺序变化 |
| `ReductionSumVector<int8 -> int32>` | Implemented + benchmarked | `RVV-native` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 常用于 row sum / cached reduction |
| `MatrixScalarMultiplyAccumulate<int8>` | Implemented + benchmarked | `RVV-native` | `Dynamic reduction helper` | Yes | `256 only` | 依赖 `RvvReduceSumInt8` |
| `MatrixBatchVectorMultiplyAccumulate<float>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 行为对齐，RVV 实现不是逐行模仿 NEON |
| `MatrixBatchVectorMultiplyAccumulate<int8 -> float>` | Implemented + benchmarked | `Semantic parity` | `Dynamic reduction helper` | Yes | `256 only` | 包含普通量化 matvec 路径 |
| `MatrixBatchVectorMultiplyAccumulate<int8, per-channel + input_offset -> float>` | Implemented + benchmarked | `Semantic parity` | `Dynamic reduction helper` | Yes | `256 only` | 包含 row sum / per-channel scale 路径 |
| `MatrixBatchVectorMultiplyAccumulate<int8 -> int16>` | Implemented + benchmarked | `Semantic parity` | `Dynamic post-accumulate + temp buffers` | Yes | `256 only` | helper 接口与 NEON 对齐，但内部是 RVV 本地化后处理链 |
| `MatrixBatchVectorMultiplyAccumulate<int8 -> int8>` | Implemented + benchmarked | `Semantic parity` | `Dynamic post-accumulate + temp buffers` | Yes | `256 only` | 同上，偏 recurrent gate / projection helper |
| `SparseMatrixBatchVectorMultiplyAccumulate1x4<float>` | Implemented + benchmarked | `Semantic parity` | `Format-fixed block` | Yes | `256 only` | block size 固定为 4，来自 sparse layout，不是 VLEN 限制 |
| `SparseMatrixBatchVectorMultiplyAccumulate<float ledger>` | Implemented + benchmarked | `Semantic parity` | `Format-fixed block` | Yes | `256 only` | ledger layout 固定按 16 元素 block 拆成 4x4 累加 |
| `SparseMatrixBatchVectorMultiplyAccumulate<int8 -> float>` | Implemented + benchmarked | `Semantic parity` | `Format-fixed block` | Yes | `256 only` | 当前已做双累加器优化，但仍受 sparse layout 约束 |
| `SparseMatrixBatchVectorMultiplyAccumulate1x16<int8>` | Implemented + benchmarked | `Semantic parity` | `Format-fixed block` | Yes | `256 only` | block size 固定为 16，row sum 已移出 batch 循环 |
| `ApplyLayerNorm<int16>` | Implemented + benchmarked | `Strict NEON-style` | `Dynamic vsetvl` | Yes | `256 only` | second pass 量化后处理链已补成 RVV 向量路径 |
| `ApplySigmoid<int16>` | Implemented + benchmarked | `Strict NEON-style` | `Dynamic vsetvl` | Yes | `256 only` | 当前按 `gemmlowp` fixed-point 公式直接展开到 RVV，不再使用 LUT |
| `ApplyTanh<int16>` | Implemented + benchmarked | `Strict NEON-style` | `Dynamic vsetvl` | Yes | `256 only` | 当前按 `gemmlowp` fixed-point 公式直接展开到 RVV |
| `CwiseMul<int16 -> int16>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 数学语义对齐，但实现不是 NEON 逐指令镜像 |
| `CwiseMul<int16 -> int8>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvl` | Yes | `256 only` | 含精确量化乘子链 |
| `CwiseAdd<int16>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 饱和语义对齐 |
| `CwiseClipping<float/int16/int8>` | Implemented + benchmarked | `RVV-native` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 更适合用 RVV 原生 min/max 写法 |
| `VectorBatchVectorCwiseProductAccumulate<int16>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvl` | Yes | `256 only` | 含精确量化乘子链 |
| `Sub1Vector<float/int16>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 接口和语义对齐，RVV 实现直接化 |
| `VectorScalarMultiply<int8 -> float>` | Implemented + benchmarked | `RVV-native` | `Dynamic vsetvl` | Yes | `256 only` | 属于通用 dequant helper |
| `MeanStddevNormalization<float>` | Implemented + benchmarked | `Semantic parity` | `Dynamic vsetvlmax + tail` | Yes | `256 only` | 行为对齐，内部按 RVV reduction + normalize 实现 |

## Practical Conclusion

| Question | Short answer |
| --- | --- |
| `rvv_tensor_utils` 的所有 helper 是不是都“照着 NEON 来了” | 不是。当前只有一部分，尤其 `ApplyLayerNorm` / `ApplySigmoid` / `ApplyTanh` 是最刻意按 NEON/gemmlowp 路线对齐的；其余多数 helper 是“语义对齐 + RVV 本地化实现” |
| 当前这些 helper 能不能支持不同 VLEN | 对 `VLEN=128/256/512` 目标范围，代码设计上是支持的 |
| 现在能不能说三档 VLEN 都已经验证完 | 不能。当前完整 benchmark 和 correctness 实测只有 `VLEN=256`；`128/512` 仍属于“设计支持、待实机验证” |
