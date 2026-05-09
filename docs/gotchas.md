# Gotchas

## 1. `USE_NEON` Is Defined on x86 via NEON_2_SSE

The existing codebase defines `USE_NEON` on x86 platforms via the NEON_2_SSE compatibility layer. This means `#ifdef USE_NEON` is true on both ARM and x86. The `USE_RVV` macro is only defined on RISC-V with vector extension, so there is no conflict — but be aware that `USE_NEON` does not necessarily mean ARM.

## 2. `#elif` vs `#endif / #ifdef` Ordering

When adding `#elif defined(USE_RVV)`, it must come AFTER the `#ifdef USE_NEON` block and BEFORE the `#endif`. If the existing code has `#ifdef USE_NEON ... #endif` followed by `#ifdef __AVX2__ ... #endif`, the RVV block should be `#elif defined(USE_RVV)` before the first `#endif`, not after it.

For files that check `#ifdef __AVX2__` first (like `integer_ops/sub.h`), the RVV block should be `#elif defined(USE_RVV)` after the AVX2 block.

## 3. Signed vs Unsigned Narrowing

NEON has separate `vqmovn_s16` (signed narrow) and `vqmovun_s16` (unsigned narrow from signed). In RVV, `vnclip` is always signed narrowing. For uint8 outputs from signed int16 intermediates, you must reinterpret: narrow to int8 with `vnclip`, then `vreinterpret` to uint8.

## 4. `vsetvl` Return Type Is `size_t`, Not `int`

RVV `vsetvl` returns `size_t`. Loop counters that interact with `vl` should also be `size_t` or at least compatible. Mixing `int i` with `size_t vl` in `i += vl` works but may trigger signed/unsigned warnings on strict compilers.

## 5. Shift Intrinsics Require Unsigned Shift Amount

RVV shift intrinsics (`vsll_vv`, `vsra_vv`) require the shift amount vector to be unsigned (`vuint32mN_t`). When using signed shift values from params, reinterpret with `vreinterpret_v_i32mN_u32mN`.

## 6. The `kNeonOptimized` Kernel Type Naming

The `kNeonOptimized` enum value and `Register_XXX_NEON_OPT()` function names are misleading when used for RVV. They actually mean "use the optimized code path in `optimized_ops.h` / `integer_ops/*.h`" rather than "use NEON specifically". The optimized headers contain both NEON and RVV blocks guarded by `#ifdef USE_NEON` / `#elif defined(USE_RVV)`.

## 7. Build on Target Hardware

The target machine is `ssh 192.168.5.211` (openkylin, password: openkylin). CMake on RISC-V auto-detects `CMAKE_SYSTEM_PROCESSOR` as `riscv64`. The RVV compiler flag `-march=rv64gcv` enables vector intrinsics. The Spacemit X100 CPU supports `rv64imafdcv` plus many extensions (zba, zbb, zbc, zbs, zvbb, zvfh, etc.).

Build command:
```bash
mkdir build && cd build
cmake ../tflite -DTFLITE_ENABLE_XNNPACK=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv"
make -j8 rvv_standalone_test rvv_speedup_test
```

## 8. XNNPACK Build Is Very Slow on RISC-V

Building with `-DTFLITE_ENABLE_XNNPACK=ON` (default) takes 30+ minutes on the Spacemit X100 due to thousands of XNNPACK microkernel source files. Use `-DTFLITE_ENABLE_XNNPACK=OFF` for faster iteration when testing only the RVV kernel changes. XNNPACK has its own RISC-V optimizations that are independent of our work.

## 9. Test Framework Has Broken Path on This Repo

`-DTFLITE_KERNEL_TEST=ON` fails with a missing `schema_conversion_utils.cc` file (expected at `compiler/mlir/lite/schema/` but actually at `tflite/converter/schema/`). Workaround: add test executables directly in the main `tflite/CMakeLists.txt` instead of through the test framework. The standalone tests (`rvv_standalone_test.cc`, `rvv_speedup_test.cc`) are self-contained and don't depend on gtest.

## 10. `neon_tensor_utils.h` Is a Dispatch Layer, Not NEON-Specific

Despite its name, `neon_tensor_utils.h` is the central dispatch layer for tensor utility functions. On RVV platforms, it `#include`s `rvv_tensor_utils.h` and skips the entire NEON/Portable dispatch block. On non-RVV platforms, it uses the `NEON_OR_PORTABLE` macro as before. Don't add RVV-specific code directly to `neon_tensor_utils.h` — put it in `rvv_tensor_utils.h` instead.

## 11. `vfredusum` Is Unordered Reduction

RVV's `vfredusum` performs unordered floating-point reduction (allows reassociation). This is fine for our use cases (dot products, sums) since we don't need strict IEEE ordering. If strict ordering were needed, use `vfredosum` instead — but it serializes the reduction and is much slower.

## 12. Per-Channel Quantize in Depthwise Conv Uses Vector Shift Amounts

The `Quantize()` function in `optimized_ops.h` applies per-channel multipliers and shifts. The NEON version loads shift amounts into vector registers and uses `vrshlq_s32` (which accepts negative shifts as right shifts). The RVV version must split into separate left-shift and right-shift operations since RVV shift intrinsics only accept unsigned shift amounts.
