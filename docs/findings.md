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

## 6. `gemmlowp::FixedPoint<int16x8_t, N>` Blocks Sigmoid/Tanh Vectorization

The NEON sigmoid and tanh implementations (`NeonApplySigmoid`, `NeonApplyTanh`) use `gemmlowp::FixedPoint<int16x8_t, N>` which is a template specialization for NEON vector types. The gemmlowp library provides `logistic()` and `tanh()` functions specialized for these types. No such specialization exists for RVV types, and creating one would require reimplementing the gemmlowp fixed-point math library. The scalar `int16_t` version is used instead.

## 7. `PortableMatrixBatchVectorMultiplyAccumulate` Signature Mismatch

The Portable version with `(scratch, result, context)` parameter order differs subtly from the Neon declaration. When delegating complex int8 GEMV variants to Portable, we must match the Portable header signatures exactly. For the `scratch`-based overload, we instead redirect to the simpler RVV int8 GEMV (without scratch) since the scratch buffer is only needed for the aarch64 dotprod fast path.

## 8. Spacemit X100 VLEN=256 Characteristics

Measured on the target hardware (Spacemit X100, rv64gcv):
- VLEN=256 confirmed (processes 8 float32 / 32 int8 per vector operation)
- Float element-wise ops (Add, Mul) show ~3x speedup — bandwidth-bound at this VLEN
- Compute-heavy ops (MatVec, MeanStddevNorm, ReductionSum) show 9-12x speedup — compute-bound, benefits from `vfmacc`/`vfredusum`
- Quantized int8 ops show 6-9x speedup — widening multiply chain benefits from RVV's native `vwmul`/`vsext`
- Sub-microsecond workloads (small MaxPool) show no speedup due to timer noise

## 9. `neon_tensor_utils.cc` Size vs `rvv_tensor_utils.cc`

847 vs 2806 lines. The gap comes from:
- 557 lines of `#ifdef __aarch64__` inline assembly (dotprod GEMV, ShuffleVectors) — no RVV port needed
- ~200 lines of NEON manual loop unrolling (`vget_low`/`vget_high` splitting) — eliminated by RVV strip-mining
- ~100 lines of helper functions (`AccumulateNeonLane`, `RoundDownVectors`) — replaced by `vredsum`/`vfredusum`
- Sigmoid/Tanh/complex-GEMV delegate to Portable (see decisions #9, #10)

## 10. CMake Source Filter Required for Test Files

The `populate_tflite_source_vars("kernels" ...)` macro auto-discovers all `.cc` files. Test and benchmark files (`rvv_*_test.cc`, `rvv_benchmark.cc`, `rvv_speedup_test.cc`) placed in `tflite/kernels/` were accidentally compiled into the library, causing link errors (duplicate `main`, missing benchmark headers). Fixed by extending the filter regex to exclude `rvv_.*_test|rvv_benchmark|rvv_speedup`.
