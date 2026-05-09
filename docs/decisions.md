# Design Decisions

## 1. RVV Detection Mechanism

**Decision**: Use a centralized `rvv_check.h` header that defines `USE_RVV` when `__riscv_v` or `__riscv_vector` is detected, included transitively through `neon_check.h`.

**Rationale**: Mirrors the existing `neon_check.h` pattern. By including `rvv_check.h` from `neon_check.h`, every file that already checks for NEON automatically gains RVV detection without adding new includes.

**Alternatives considered**: Defining `USE_RVV` via CMake `-D` flag only. Rejected because the compiler-defined `__riscv_v` / `__riscv_vector` macros are the authoritative source of RVV availability.

## 2. VLEN-Agnostic Strip-Mining Pattern

**Decision**: All RVV loops use `__riscv_vsetvl_eXXmN(remaining)` at loop top and advance by the returned `vl`. No hardcoded vector widths.

**Rationale**: The challenge mandates VLEN 128/256/512 adaptive support. Strip-mining with `vsetvl` handles arbitrary VLEN, including odd remainders, without a scalar tail loop.

**NEON contrast**: NEON uses fixed `i += 16` loops with separate `i += 8` and `i += 4` cleanup loops plus a final scalar loop. RVV's `vsetvl` unifies all these into a single loop — this is why `rvv_tensor_utils.cc` (847 lines) is much shorter than `neon_tensor_utils.cc` (2806 lines) despite covering the same functions.

## 3. LMUL Selection

**Decision**: Use m4 for int8 element-wise ops (widening chain: int8m1 -> int16m2 -> int32m4), m2 for int16 ops (int16m1 -> int32m2), m1 for standalone int32/float ops. Use m4 for float element-wise for throughput.

**Rationale**: The widening chain from int8 to int32 requires 4x the register width. Starting at m1 for the narrowest type and widening naturally fills m4 for the widest, which is the maximum that allows further widening to m8. This balances register pressure against throughput.

## 4. Placement of RVV Blocks in Operator Headers

**Decision**: Use `#elif defined(USE_RVV)` after `#ifdef USE_NEON` blocks rather than independent `#ifdef USE_RVV` blocks.

**Rationale**: The `#elif` pattern ensures mutual exclusion — NEON and RVV are never both active (which is physically impossible anyway). It also means the RVV path processes all elements, so the trailing scalar loop is naturally skipped (since `i == size` after the RVV loop).

## 5. Quantized Multiply Approach

**Decision**: Implement `SaturatingRoundingDoublingHighMul` as widening multiply to int64 + add nudge + shift right 31, using `vwmul` + `vadd` + `vsra` + `vnclip`.

**Rationale**: NEON has a single-instruction `vqrdmulhq_n_s32` for this operation. RVV lacks an equivalent, so we must decompose it. The widening approach gives exact results matching the NEON semantics.

## 6. Depthwise Conv Strategy

**Decision**: Optimize the generic accumulation row function (`QuantizedDepthwiseConvAccumRowGeneric` and `FloatDepthwiseConvAccumRowGeneric`) with RVV for the common `depth_multiplier == 1` case, falling back to scalar for other depth multipliers.

**Rationale**: The NEON depthwise conv uses template specializations for specific (input_depth, depth_multiplier) combinations that are deeply tied to 128-bit register geometry. Instead of creating many RVV specializations, a single VLEN-agnostic implementation for `depth_multiplier == 1` covers the most common use case. The template specializations are automatically skipped on non-NEON platforms anyway (wrapped in `#ifdef USE_NEON`).

## 7. Kernel Registration Reuse

**Decision**: Reuse `Register_XXX_NEON_OPT()` for RVV by changing `#ifdef USE_NEON` to `#if defined(USE_NEON) || defined(USE_RVV)` in the dispatch functions.

**Rationale**: The "NEON_OPT" variant name is misleading but the actual effect is simply to route through the `optimized_integer_ops` namespace, which contains both NEON and RVV code paths guarded by their respective `#ifdef`s. Creating separate `Register_XXX_RVV_OPT()` functions would duplicate code for no benefit.

## 8. Tensor Utils: Separate `rvv_tensor_utils.h` Dispatch Header

**Decision**: Create a standalone `rvv_tensor_utils.h` with direct `RvvXxx()` calls, and have `neon_tensor_utils.h` redirect to it via `#ifdef USE_RVV` at the top.

**Rationale**: `neon_tensor_utils.h` is actually a dispatch layer that uses the `NEON_OR_PORTABLE` macro. Rather than relying on the macro for RVV dispatch, we create a clean parallel header (`rvv_tensor_utils.h`) that calls `RvvXxx()` directly. This avoids modifying any NEON logic and keeps the RVV dispatch self-contained.

**Structure**:
```
neon_tensor_utils.h
├── #ifdef USE_RVV → includes rvv_tensor_utils.h
│   ├── rvv_tensor_utils_impl.h (RVV declarations)
│   └── portable_tensor_utils_impl.h (fallback)
└── #else → original NEON_OR_PORTABLE dispatch
    ├── neon_tensor_utils_impl.h
    └── portable_tensor_utils_impl.h
```

## 9. Sigmoid/Tanh Remain on Portable

**Decision**: `RvvApplySigmoid` and `RvvApplyTanh` delegate to `PortableApplySigmoid` / `PortableApplyTanh` instead of implementing RVV versions.

**Rationale**: The NEON implementations use `gemmlowp::FixedPoint<int16x8_t, N>` template specializations for vectorized fixed-point logistic/tanh lookup tables. These specializations only exist for NEON vector types (`int16x8_t`). There is no gemmlowp specialization for RVV types. Implementing a custom RVV LUT-based sigmoid/tanh would require reimplementing the gemmlowp fixed-point math library for RVV, which is out of scope.

## 10. aarch64 Inline Assembly Not Ported

**Decision**: The ~557 lines of `#ifdef __aarch64__` code in `neon_tensor_utils.cc` (dotprod GEMV, ShuffleVectors, sdot detection) are not ported to RVV.

**Rationale**: This code uses ARM-specific instructions (`sdot`, `smmla`, `st4`) and register conventions via inline assembly. The equivalent RVV approach is different — RVV's `vwmul`/`vwmacc` handle the widening multiply-accumulate directly without needing data shuffling. The RVV int8 GEMV implementation uses intrinsics-based dot products instead. Complex variants that need Ruy/GEMM backends delegate to Portable.
