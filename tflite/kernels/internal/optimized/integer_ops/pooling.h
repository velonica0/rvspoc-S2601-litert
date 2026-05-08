/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_

#include <string.h>

#include <algorithm>
#include <limits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/cppmath.h"
#include "tflite/kernels/internal/optimized/cpu_check.h"
#include "tflite/kernels/internal/optimized/im2col_utils.h"
#include "tflite/kernels/internal/optimized/neon_check.h"
#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/kernels/internal/quantization_util.h"
#include "tflite/kernels/internal/reference/reference_ops.h"
#include "tflite/kernels/internal/strided_slice_logic.h"
#include "tflite/kernels/internal/tensor_utils.h"
#include "tflite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

#ifdef USE_RVV
inline vint16m1_t RvvLoadInt8AsInt16ForPooling(const int8_t* data, size_t vl) {
  return __riscv_vwadd_vx_i16m1(__riscv_vle8_v_i8mf2(data, vl), 0, vl);
}

inline void RvvStoreInt8FromInt32ForPooling(vint32m2_t values, int8_t* data,
                                            size_t vl) {
  const vint16m1_t narrowed16 = __riscv_vnsra_wx_i16m1(values, 0, vl);
  const vint8mf2_t narrowed8 = __riscv_vnsra_wx_i8mf2(narrowed16, 0, vl);
  __riscv_vse8_v_i8mf2(data, narrowed8, vl);
}

inline vuint8m1_t RvvLoadBiasedInt8AsUInt8ForPooling(const int8_t* data,
                                                     size_t vl) {
  const vint8m1_t values = __riscv_vle8_v_i8m1(data, vl);
  const vuint8m1_t bits = __riscv_vreinterpret_v_i8m1_u8m1(values);
  return __riscv_vxor_vx_u8m1(bits, 0x80, vl);
}

inline void RvvStoreBiasedUInt8AsInt8ForPooling(vuint8m1_t values, int8_t* data,
                                                size_t vl) {
  const vuint8m1_t unbiased = __riscv_vxor_vx_u8m1(values, 0x80, vl);
  __riscv_vse8_v_i8m1(data, __riscv_vreinterpret_v_u8m1_i8m1(unbiased), vl);
}

inline vint32m2_t RvvAverageRoundAwayFromZeroForPooling(vint32m2_t values,
                                                        int filter_count,
                                                        size_t vl) {
  const vbool16_t negative = __riscv_vmslt_vx_i32m2_b16(values, 0, vl);
  const vint32m2_t negated = __riscv_vneg_v_i32m2(values, vl);
  const vint32m2_t abs_values =
      __riscv_vmerge_vvm_i32m2(values, negated, negative, vl);
  const vuint32m2_t abs_values_u32 =
      __riscv_vreinterpret_v_i32m2_u32m2(abs_values);
  vuint32m2_t rounded_abs_u32;
  if (filter_count == 4) {
    rounded_abs_u32 = __riscv_vsrl_vx_u32m2(
        __riscv_vadd_vx_u32m2(abs_values_u32, 2, vl), 2, vl);
  } else if (filter_count == 9) {
    rounded_abs_u32 = __riscv_vsrl_vx_u32m2(
        __riscv_vadd_vx_u32m2(
            __riscv_vmul_vx_u32m2(abs_values_u32, 7282u, vl), 1u << 15, vl),
        16, vl);
  } else if (filter_count == 15) {
    rounded_abs_u32 = __riscv_vsrl_vx_u32m2(
        __riscv_vadd_vx_u32m2(
            __riscv_vmul_vx_u32m2(abs_values_u32, 4369u, vl), 1u << 15, vl),
        16, vl);
  } else {
    const vuint32m2_t quotient =
        __riscv_vdivu_vx_u32m2(abs_values_u32, filter_count, vl);
    const vuint32m2_t remainder =
        __riscv_vremu_vx_u32m2(abs_values_u32, filter_count, vl);
    const vuint32m2_t doubled_remainder =
        __riscv_vsll_vx_u32m2(remainder, 1, vl);
    const vbool16_t increment_mask =
        __riscv_vmsgeu_vx_u32m2_b16(doubled_remainder, filter_count, vl);
    const vuint32m2_t increment = __riscv_vmerge_vxm_u32m2(
        __riscv_vmv_v_x_u32m2(0, vl), 1, increment_mask, vl);
    rounded_abs_u32 = __riscv_vadd_vv_u32m2(quotient, increment, vl);
  }
  const vint32m2_t rounded_abs =
      __riscv_vreinterpret_v_u32m2_i32m2(rounded_abs_u32);
  const vint32m2_t negative_rounded_abs = __riscv_vneg_v_i32m2(rounded_abs, vl);
  return __riscv_vmerge_vvm_i32m2(rounded_abs, negative_rounded_abs, negative,
                                  vl);
}
#endif

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8_t* input_data, const RuntimeShape& output_shape,
                    int8_t* output_data) {
  ruy::profiler::ScopeLabel label("MaxPool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int8_t acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
#ifdef USE_RVV
          // Keep RVV max reduction on biased u8 values so we can reuse the
          // faster unsigned compare path for signed int8 ordering.
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
#else
          memset(acc, params.quantized_activation_min,
                 tranche_depth * sizeof(acc[0]));
#endif
          const int8_t* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8_t* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8_t* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                int8x16_t acc_reg = vld1q_s8(acc + channel);
                int8x16_t input_reg = vld1q_s8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg = vmaxq_s8(acc_reg, input_reg);
                vst1q_s8(acc + channel, acc_reg);
              }

              for (; channel <= tranche_depth - 8; channel += 8) {
                int8x8_t acc_reg = vld1_s8(acc + channel);
                int8x8_t input_reg = vld1_s8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vmax_s8(acc_reg, input_reg);
                vst1_s8(acc + channel, acc_reg);
              }
#endif
#ifdef USE_RVV
              for (; channel < tranche_depth;) {
                const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
                vuint8m1_t acc_reg =
                    __riscv_vle8_v_u8m1(reinterpret_cast<uint8_t*>(acc) + channel,
                                        vl);
                const vuint8m1_t input_reg =
                    RvvLoadBiasedInt8AsUInt8ForPooling(input_channel_ptr, vl);
                acc_reg = __riscv_vmaxu_vv_u8m1(acc_reg, input_reg, vl);
                __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(acc) + channel,
                                    acc_reg, vl);
                input_channel_ptr += vl;
                channel += static_cast<int>(vl);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] = std::max(acc[channel], *input_channel_ptr++);
              }
              input_row_ptr += depth;
            }
          }
          int8_t* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                    out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 16; channel += 16) {
            int8x16_t a = vld1q_s8(acc + channel);
            a = vminq_s8(a, vdupq_n_s8(params.quantized_activation_max));
            a = vmaxq_s8(a, vdupq_n_s8(params.quantized_activation_min));
            vst1q_s8(output_ptr + channel, a);
          }
          for (; channel <= tranche_depth - 8; channel += 8) {
            int8x8_t a = vld1_s8(acc + channel);
            a = vmin_s8(a, vdup_n_s8(params.quantized_activation_max));
            a = vmax_s8(a, vdup_n_s8(params.quantized_activation_min));
            vst1_s8(output_ptr + channel, a);
          }
#endif
#ifdef USE_RVV
          const bool full_int8_range =
              params.quantized_activation_min ==
                  std::numeric_limits<int8_t>::min() &&
              params.quantized_activation_max ==
                  std::numeric_limits<int8_t>::max();
          if (full_int8_range) {
            for (; channel < tranche_depth;) {
              const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
              const vuint8m1_t values = __riscv_vle8_v_u8m1(
                  reinterpret_cast<uint8_t*>(acc) + channel, vl);
              RvvStoreBiasedUInt8AsInt8ForPooling(values, output_ptr + channel,
                                                  vl);
              channel += static_cast<int>(vl);
            }
          } else {
            for (; channel < tranche_depth;) {
              const size_t vl = __riscv_vsetvl_e8m1(tranche_depth - channel);
              const vuint8m1_t biased_values = __riscv_vle8_v_u8m1(
                  reinterpret_cast<uint8_t*>(acc) + channel, vl);
              vint8m1_t values = __riscv_vreinterpret_v_u8m1_i8m1(
                  __riscv_vxor_vx_u8m1(biased_values, 0x80, vl));
              values = __riscv_vmin_vx_i8m1(values,
                                            params.quantized_activation_max, vl);
              values = __riscv_vmax_vx_i8m1(values,
                                            params.quantized_activation_min, vl);
              __riscv_vse8_v_i8m1(output_ptr + channel, values, vl);
              channel += static_cast<int>(vl);
            }
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            int8_t a = acc[channel];
            a = std::max<int8_t>(a, params.quantized_activation_min);
            a = std::min<int8_t>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<int8_t>(a);
          }
        }
      }
    }
  }
}

inline bool AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const RuntimeShape& output_shape, int8_t* output_data) {
  ruy::profiler::ScopeLabel label("AveragePool/8bitWith32bitAccumulator");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int32_t acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          if (filter_count == 0) return false;
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const int8_t* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const int8_t* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const int8_t* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                int16x4_t acc_reg[4];
                int8x16_t input_reg = vld1q_s8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg[0] = vget_low_s16(vmovl_s8(vget_low_s8(input_reg)));
                acc_reg[1] = vget_high_s16(vmovl_s8(vget_low_s8(input_reg)));
                acc_reg[2] = vget_low_s16(vmovl_s8(vget_high_s8(input_reg)));
                acc_reg[3] = vget_high_s16(vmovl_s8(vget_high_s8(input_reg)));
                for (int i = 0; i < 4; i++) {
                  vst1q_s32(
                      acc + channel + 4 * i,
                      vaddw_s16(vld1q_s32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
              for (; channel <= tranche_depth - 8; channel += 8) {
                int16x4_t acc_reg[2];
                int16x8_t input_reg = vmovl_s8(vld1_s8(input_channel_ptr));
                input_channel_ptr += 8;
                acc_reg[0] = vget_low_s16(input_reg);
                acc_reg[1] = vget_high_s16(input_reg);
                for (int i = 0; i < 2; i++) {
                  vst1q_s32(
                      acc + channel + 4 * i,
                      vaddw_s16(vld1q_s32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
#endif
#ifdef USE_RVV
              for (; channel < tranche_depth;) {
                const size_t vl = __riscv_vsetvl_e16m1(tranche_depth - channel);
                const vint16m1_t input_values =
                    RvvLoadInt8AsInt16ForPooling(input_channel_ptr, vl);
                vint32m2_t acc_values = __riscv_vle32_v_i32m2(acc + channel, vl);
                acc_values =
                    __riscv_vadd_vv_i32m2(acc_values,
                                          __riscv_vwadd_vx_i32m2(input_values, 0,
                                                                 vl),
                                          vl);
                __riscv_vse32_v_i32m2(acc + channel, acc_values, vl);
                input_channel_ptr += vl;
                channel += static_cast<int>(vl);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] += *input_channel_ptr++;
              }
              input_row_ptr += depth;
            }
          }
          int8_t* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                    out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 8; channel += 8) {
            int16_t buf[8];
            for (int i = 0; i < 8; i++) {
              buf[i] =
                  acc[channel + i] > 0
                      ? (acc[channel + i] + filter_count / 2) / filter_count
                      : (acc[channel + i] - filter_count / 2) / filter_count;
            }
            int8x8_t buf8 = vqmovn_s16(vld1q_s16(buf));
            buf8 = vmin_s8(buf8, vdup_n_s8(params.quantized_activation_max));
            buf8 = vmax_s8(buf8, vdup_n_s8(params.quantized_activation_min));
            vst1_s8(output_ptr + channel, buf8);
          }
#endif
#ifdef USE_RVV
          for (; channel < tranche_depth;) {
            const size_t vl = __riscv_vsetvl_e32m2(tranche_depth - channel);
            vint32m2_t averaged = RvvAverageRoundAwayFromZeroForPooling(
                __riscv_vle32_v_i32m2(acc + channel, vl), filter_count, vl);
            averaged = __riscv_vmax_vx_i32m2(averaged,
                                             params.quantized_activation_min,
                                             vl);
            averaged = __riscv_vmin_vx_i32m2(averaged,
                                             params.quantized_activation_max,
                                             vl);
            RvvStoreInt8FromInt32ForPooling(averaged, output_ptr + channel, vl);
            channel += static_cast<int>(vl);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            int16_t a = acc[channel] > 0
                            ? (acc[channel] + filter_count / 2) / filter_count
                            : (acc[channel] - filter_count / 2) / filter_count;
            a = std::max<int16_t>(a, params.quantized_activation_min);
            a = std::min<int16_t>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<int8_t>(a);
          }
        }
      }
    }
  }
  return true;
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_POOLING_H_
