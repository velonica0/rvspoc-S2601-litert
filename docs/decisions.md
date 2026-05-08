# RVV Decisions

- 2026-05-06: 第一批 RISC-V/RVV 优化先落在 `tflite/kernels/internal` 的 float `MatrixBatchVectorMultiplyAccumulate`，因为它已经有现成单测和 benchmark，便于和 scalar 基线直接对比。
- 2026-05-06: RVV 分发改成仿照 NEON/SSE 的组织方式：`reference/portable_tensor_utils.h` 保持纯 portable 实现，`optimized/rvv_tensor_utils.h` 负责 RVV wrapper，再由 `tflite/kernels/internal/tensor_utils.cc` 依据 `USE_RVV` 选路，避免把 RVV 分发逻辑继续堆进 `reference` 头文件。
- 2026-05-06: 构建侧显式启用 `-march=rv64gcv_zvl128b -mabi=lp64d`，同时实现用 `vsetvl`/`vsetvlmax` 按运行时 VLEN 自适应，覆盖 VLEN 128/256/512。
- 2026-05-06: “所有 kernel” 的实施目标定义为“全部 CPU kernel 在 RISC-V 上可运行，并对比赛模型热点 kernel 做 RVV 加速”，而不是给 300+ 个上层算子逐个重写一份 RVV 专核。
- 2026-05-06: 第二批 RVV 覆盖优先扩展共享 helper，而不是继续堆单个 top-level kernel；本轮新增的入口包括 float `VectorVectorDotProduct` / `ReductionSumVector`，int8 `ReductionSumVector` / `MatrixScalarMultiplyAccumulate` / `MatrixBatchVectorMultiplyAccumulate`（含 `per_channel_scale + input_offset` 路径）。
- 2026-05-06: helper benchmark 继续直接编译 `reference/portable_tensor_utils.cc` 与 `optimized/rvv_tensor_utils.cc` 做 scalar/RVV 对比；而 public helper 的真实分发正确性则单独通过 `tensor_utils.cc` 的语法检查和 dispatch probe 验证，两者分开记录，避免 benchmark 掩盖接线问题。
- 2026-05-06: benchmark 的精度门槛以 `max abs diff` 和 `mean abs diff` 为主，`max rel diff` 只作为辅助手段，因为接近零输出时相对误差会被不成比例地放大。
- 2026-05-08: `reduce.h` / `resize_bilinear.h` 这批 follow-up P1 operator benchmark 继续按“按源文件拆分 `.inc` section”组织，不再把新增算子继续堆进单一 benchmark 文件，便于后续把 benchmark 结果和算子文件直接对应。
- 2026-05-08: 新增的 `Mean<float>` 与 `ResizeBilinear<float>` benchmark 精度校验对齐 `CODEX.md` 的 FP32 目标，采用 `max_abs_diff <= 1e-5` 或 `max_rel_diff <= 1e-5 且 mean_abs_diff <= 1e-6` 的组合门槛，而不是复用此前更激进的 `1e-6` 通用阈值。
