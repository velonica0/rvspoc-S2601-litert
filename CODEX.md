 CLAUDE.md — RVSPOC S2601 (LiteRT RISC-V + RVV)


## 仓库结构

本仓库 = LiteRT 完整源码（566 MB，~7800 文件）。我们的工作以 patch 方式叠加在上面，不动 LiteRT 现有文件结构（除非必要修改 `#ifdef` 分支或 BUILD 配置）。

新增 RVV 内核统一放在原文件相邻位置或 `tflite/kernels/internal/optimized/rvv/` 子目录（实施时再拍板）。

## 开发约束（来自赛题，硬性）

- C++17 或以上
- RVV 实现优先用 intrinsics（`<riscv_vector.h>`），可辅以内联汇编
- **必须**保留 scalar 回退（条件编译 `#ifdef __riscv_vector` 隔离）
- 代码风格：Google C++ Style Guide
- 必须支持 VLEN 128/256/512 自适应（用 `vsetvl`）
- 构建：将代码同步至K3(ssh连接192.168.5.103)/home/openkylin/github/rvspoc-S2601-litert目录下 用户名密码都是openkylin 随后在K3中编译
- License：Apache 2.0（与 LiteRT 一致）
- **禁止**搬运现有第三方 RISC-V TFLite 移植，自主实现或明确标注引用
- 用 AI 辅助写代码需在最终报告中说明使用方式及占比

## 精度与性能门槛

| 项 | 标准 |
|---|---|
| FP32 模型 Top-1 vs x86 | ≤ 0.1% |
| INT8 模型 Top-1 vs x86 | ≤ 1% |
| FP32 算子相对误差 | ≤ 1e-5 |
| INT8 算子差值 | ≤ 1 LSB |
| 推理延迟 | ≤ 110 ms（avg/p50/p95） |

## 提交规范

- commit message 详细写明改动 + why + 影响面（参考全局 CLAUDE.md 第 4 条）
- 决策类信息记 `docs/decisions.md`，调试发现记 `docs/findings.md`，踩坑记 `docs/gotchas.md`
- 每个新增/修改的 RVV 内核必须有：
  - 配套精度测试（与 scalar 对比）
  - 配套 benchmark（与 scalar 对比）
  - commit message 说明算子、VLEN 假设、测试条件

## 不要做的事

- 不要改动 LiteRT 上游已有 ARM Neon 代码的逻辑（只读不改，作为参考）
- 不要混用 schema 的 push/migrate 模式（本项目无此问题，但保持警觉）
- 不要在 PR 标题/描述/公开文档暴露：内网 IP、个人信息、未脱敏的客户名（参考全局 CLAUDE.md 第 6 条）

## 任务结束收尾格式

照全局 CLAUDE.md 第 8 条，每个任务末尾给：
```
改动：...
验证：...
遗留：...
```