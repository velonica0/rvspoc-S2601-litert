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

The target machine is `ssh 192.168.5.211` (openkylin). CMake on RISC-V should auto-detect `CMAKE_SYSTEM_PROCESSOR` as `riscv64`. The RVV compiler flag `-march=rv64gcv` enables vector intrinsics. Some toolchains may need `-march=rv64gcv_zba_zbb` for additional extensions.
