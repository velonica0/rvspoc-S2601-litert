# Debugging Findings

## 1. `vnclip` Requires Rounding Mode Argument

RVV 1.0 intrinsics for `vnclip` require an explicit rounding mode parameter (`__RISCV_VXRM_RDN`, `__RISCV_VXRM_RNU`, etc.). Earlier RVV draft specs did not require this. When narrowing from int32 to int16 or int16 to int8 after clamping, use `__RISCV_VXRM_RDN` (round-down / truncate) since the values are already clamped to the target range.

## 2. Widening Chain Register Pressure

The int8 quantized ops widen through three levels: int8m1 -> int16m2 -> int32m4 -> int64m8 (for the multiply-high operation). At m8 level, all 32 vector registers are consumed by a single vector, leaving no room for temporaries. Solution: perform the SRDH (saturating rounding doubling high multiply) operation at the m4 level, widening to m8 only for the intermediate 64-bit product, then immediately narrowing back to m4.

## 3. NEON `vqrdmulhq_n_s32` Has No Direct RVV Equivalent

This NEON instruction performs saturating rounding doubling high multiply in a single instruction. The RVV decomposition requires:
1. `vwmul_vx` — widening multiply to 64-bit
2. `vadd` with nudge `(1 << 30)` — rounding
3. `vsra` by 31 — extract high 32 bits
4. `vnclip` — saturating narrow back to 32-bit

This is 4 instructions vs 1, but RVV's wider vectors compensate by processing more elements per iteration.

## 4. `gemmlowp::RoundingDivideByPOT` Overloads for NEON Types

The NEON code uses `gemmlowp::RoundingDivideByPOT` which has overloads for NEON vector types (using `vrshlq_s32`). For RVV, we implement our own `rvv_utils::RoundingDivideByPOT` that matches the gemmlowp semantics: `(x + ((1 << exp) >> 1) + (sign_bit & ((1 << exp) - 1))) >> exp`.

## 5. Depthwise Conv Template Specializations Are NEON-Specific

The NEON depthwise conv has ~15 template specializations for specific `(input_depth, depth_multiplier)` pairs (e.g., `(8,2)`, `(2,1)`, `(1,32)`). Each uses NEON register geometry assumptions (e.g., processing exactly 8 channels at once). These cannot be directly ported to RVV without creating VLEN-dependent specializations, which contradicts the VLEN-agnostic requirement. The generic accumulation function is the correct RVV target.
