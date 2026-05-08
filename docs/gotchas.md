# RVV Gotchas

- `__riscv_vector` 不是 RISC-V 主机默认就会定义的宏，不能只依赖 `riscv64` 架构判断，必须配合 RVV 编译选项。
- 这轮实现保持 portable 回退，任何 RVV 入口都必须放在 `#ifdef __riscv_vector` 保护内，避免 x86/ARM 构建误入 `<riscv_vector.h>`。
- `tflite/kernels/internal/portable_tensor_utils.h` 和 `tflite/kernels/internal/reference/portable_tensor_utils.h` 文件名相同；远端同步时如果把两者一起拷到同一个目标目录，顶层 portable header 会被 `reference` 版本覆盖，随后 `tensor_utils.cc` 会出现大批重定义和声明缺失错误。同步这两个文件时必须显式写到各自完整目标路径。
- K3 环境里没有 `rg`，远端排查时优先用 `grep`/`find`，避免把工具缺失误判成源码问题。
- GCC 15 的 `riscv_vector.h` 不适合靠文本搜索猜 intrinsic，实际排查效率最高的方式是“最小 probe 源文件 + K3 上 `g++ -fsyntax-only`”。
- `tools/riscv/run_rvv_tensor_utils_benchmark.sh` 直接编译 `optimized/rvv_tensor_utils.cc` 做数值与性能对比，所以 benchmark 通过并不自动代表 `tensor_utils.cc` 这条 public helper 分发链也已经接好；结构验证仍然要补 `-fsyntax-only` 和 dispatch probe。
- `optimized_ops` 这类 operator benchmark 不能直接复用 `tensor_utils` benchmark 那套最小 include 集；在 K3 上至少要补上 `litert/cmake_build/eigen`、`litert/cmake_build/ruy`、`litert/cmake_build/gemmlowp`，而且链接时还要把 `tflite/kernels/internal/runtime_shape.cc` 一起编进去，否则会在 `RuntimeShape::{Dims,FlatSize,~RuntimeShape}` 这里报 unresolved symbol。
- 如果把 scalar/RVV 两套 header-inline kernel 一起链接进同一个 benchmark binary，要特别警惕 weak symbol 串台。`BinaryBroadcastFiveFold` 这种“模板外层 + 函数指针 inner kernel”的写法，在 `AddScalarBroadcast` / `MulSimpleBroadcast` 这类 inline 符号上会让 RVV 侧错误绑回 scalar 实现，表现就是 benchmark 看起来只有 `~1.00x`。规避方式有两种：要么拆成两个可执行文件分别测，要么像现在这样在 benchmark TU 里放本地 internal-linkage wrapper，再把函数指针指向本地 wrapper。
- 浮点 matvec/reduction 在大 shape 上会出现由累加顺序改变带来的微小差异；当输出本身接近零时，`max rel diff` 很容易失真，不能单独拿来判错。
