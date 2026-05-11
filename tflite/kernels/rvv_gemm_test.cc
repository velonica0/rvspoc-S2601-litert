// Minimal RVV GEMM correctness test.
#include <cstdio>
#include <cmath>
#include <cstring>
#include "tflite/kernels/internal/optimized/rvv_check.h"

#ifdef USE_RVV
#include <riscv_vector.h>

void rvv_gemm_float(const float* lhs, const float* rhs, float* dst,
                    int rows, int depth, int cols) {
  for (int col = 0; col < cols; col++) {
    const float* rhs_col = rhs + col * depth;
    for (int row = 0; row < rows; row++) {
      const float* lhs_row = lhs + row * depth;
      size_t vl;
      vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f,
                                                   __riscv_vsetvl_e32m4(depth));
      for (int k = 0; k < depth; k += vl) {
        vl = __riscv_vsetvl_e32m4(depth - k);
        vfloat32m4_t a = __riscv_vle32_v_f32m4(lhs_row + k, vl);
        vfloat32m4_t b = __riscv_vle32_v_f32m4(rhs_col + k, vl);
        acc = __riscv_vfmacc_vv_f32m4(acc, a, b, vl);
      }
      vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
      vfloat32m1_t s = __riscv_vfredusum_vs_f32m4_f32m1(
          acc, zero, __riscv_vsetvl_e32m4(depth));
      dst[col * rows + row] = __riscv_vfmv_f_s_f32m1_f32(s);
    }
  }
}
#endif

void scalar_gemm_float(const float* lhs, const float* rhs, float* dst,
                       int rows, int depth, int cols) {
  for (int col = 0; col < cols; col++) {
    for (int row = 0; row < rows; row++) {
      float sum = 0;
      for (int k = 0; k < depth; k++)
        sum += lhs[row * depth + k] * rhs[col * depth + k];
      dst[col * rows + row] = sum;
    }
  }
}

int main() {
  // Test: A[4x8] * B[8x3] = C[4x3]
  int rows = 4, depth = 8, cols = 3;
  float A[32], B[24], C_scalar[12], C_rvv[12];

  for (int i = 0; i < 32; i++) A[i] = (float)(i % 7) - 3.0f;
  for (int i = 0; i < 24; i++) B[i] = (float)(i % 5) - 2.0f;

  scalar_gemm_float(A, B, C_scalar, rows, depth, cols);
#ifdef USE_RVV
  rvv_gemm_float(A, B, C_rvv, rows, depth, cols);
  printf("Test 4x8 * 8x3:\n");
  float max_err = 0;
  for (int i = 0; i < 12; i++) {
    float err = fabs(C_scalar[i] - C_rvv[i]);
    if (err > max_err) max_err = err;
    if (err > 0.01f)
      printf("  C[%d]: scalar=%.4f rvv=%.4f err=%.4f\n", i, C_scalar[i], C_rvv[i], err);
  }
  printf("  max_err=%.6f %s\n", max_err, max_err < 0.001f ? "PASS" : "FAIL");

  // Larger test
  int R = 64, D = 256, Co = 32;
  float* LA = new float[R * D];
  float* LB = new float[D * Co];
  float* LC_s = new float[R * Co];
  float* LC_r = new float[R * Co];
  for (int i = 0; i < R * D; i++) LA[i] = (float)((i * 7 + 3) % 19) - 9.0f;
  for (int i = 0; i < D * Co; i++) LB[i] = (float)((i * 11 + 5) % 13) - 6.0f;
  scalar_gemm_float(LA, LB, LC_s, R, D, Co);
  rvv_gemm_float(LA, LB, LC_r, R, D, Co);
  max_err = 0;
  int mismatches = 0;
  for (int i = 0; i < R * Co; i++) {
    float err = fabs(LC_s[i] - LC_r[i]);
    if (err > max_err) max_err = err;
    if (err > 1.0f) mismatches++;
  }
  printf("\nTest 64x256 * 256x32:\n");
  printf("  max_err=%.6f mismatches=%d %s\n", max_err, mismatches,
         max_err < 1.0f ? "PASS" : "FAIL");
  delete[] LA; delete[] LB; delete[] LC_s; delete[] LC_r;
#else
  printf("USE_RVV not defined, skipping.\n");
#endif
  return 0;
}
