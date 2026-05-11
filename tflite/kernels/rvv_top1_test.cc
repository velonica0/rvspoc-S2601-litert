/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). */
// Top-1 accuracy comparison: RVV vs Scalar.
// Runs the same model with the same random inputs on both builds,
// compares argmax predictions.
//
// Usage: ./rvv_top1_test <model.tflite> [num_samples]
//
// This binary outputs predictions to stdout (one per line).
// Compare two builds:
//   build/rvv_top1_test model.tflite 1000 > /tmp/rvv_preds.txt
//   build-scalar/rvv_top1_test model.tflite 1000 > /tmp/scalar_preds.txt
//   diff /tmp/rvv_preds.txt /tmp/scalar_preds.txt

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/model_builder.h"
#include "tflite/kernels/register.h"

// Simple deterministic PRNG (xoshiro128+) for reproducibility across builds
static uint32_t s[4] = {0x12345678, 0x9abcdef0, 0x13579bdf, 0x2468ace0};
static uint32_t xoshiro128p() {
  uint32_t result = s[0] + s[3];
  uint32_t t = s[1] << 9;
  s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
  s[2] ^= t;
  s[3] = (s[3] << 11) | (s[3] >> 21);
  return result;
}
static float rand_float() {
  return (float)(xoshiro128p() & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
}
static int8_t rand_int8() {
  return (int8_t)(xoshiro128p() & 0xFF);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.tflite> [num_samples] [num_threads]\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const int num_samples = argc > 2 ? atoi(argv[2]) : 100;
  const int num_threads = argc > 3 ? atoi(argv[3]) : 1;

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!model) { fprintf(stderr, "Failed to load: %s\n", model_path); return 1; }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) { fprintf(stderr, "Failed to build interpreter\n"); return 1; }

  interpreter->SetNumThreads(num_threads);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "Failed to allocate tensors\n"); return 1;
  }

  auto* input_tensor = interpreter->input_tensor(0);
  auto* output_tensor = interpreter->output_tensor(0);
  const int input_size = input_tensor->bytes;
  const int output_elements = output_tensor->bytes /
      (output_tensor->type == kTfLiteFloat32 ? 4 :
       output_tensor->type == kTfLiteInt8 ? 1 : 1);

  // Print header to stderr (not stdout, so diff works)
  fprintf(stderr, "Model: %s\n", model_path);
  fprintf(stderr, "Input type: %d, size: %d bytes\n", input_tensor->type, input_size);
  fprintf(stderr, "Output type: %d, elements: %d\n", output_tensor->type, output_elements);
  fprintf(stderr, "Samples: %d, Threads: %d\n", num_samples, num_threads);
#ifdef USE_RVV
  fprintf(stderr, "Backend: RVV\n");
#else
  fprintf(stderr, "Backend: Scalar\n");
#endif

  for (int sample = 0; sample < num_samples; sample++) {
    // Reset PRNG to same state for this sample (deterministic per-sample)
    s[0] = 0x12345678 ^ sample;
    s[1] = 0x9abcdef0 ^ (sample * 7);
    s[2] = 0x13579bdf ^ (sample * 13);
    s[3] = 0x2468ace0 ^ (sample * 31);

    // Fill input with deterministic random data
    if (input_tensor->type == kTfLiteFloat32) {
      float* data = reinterpret_cast<float*>(input_tensor->data.raw);
      for (int i = 0; i < input_size / 4; i++) data[i] = rand_float();
    } else if (input_tensor->type == kTfLiteInt8) {
      int8_t* data = reinterpret_cast<int8_t*>(input_tensor->data.raw);
      for (int i = 0; i < input_size; i++) data[i] = rand_int8();
    } else if (input_tensor->type == kTfLiteUInt8) {
      uint8_t* data = reinterpret_cast<uint8_t*>(input_tensor->data.raw);
      for (int i = 0; i < input_size; i++) data[i] = (uint8_t)(xoshiro128p() & 0xFF);
    }

    interpreter->Invoke();

    // Find argmax
    int top1 = 0;
    if (output_tensor->type == kTfLiteFloat32) {
      const float* out = reinterpret_cast<const float*>(output_tensor->data.raw);
      float max_val = out[0];
      for (int i = 1; i < output_elements; i++) {
        if (out[i] > max_val) { max_val = out[i]; top1 = i; }
      }
    } else if (output_tensor->type == kTfLiteInt8) {
      const int8_t* out = reinterpret_cast<const int8_t*>(output_tensor->data.raw);
      int8_t max_val = out[0];
      for (int i = 1; i < output_elements; i++) {
        if (out[i] > max_val) { max_val = out[i]; top1 = i; }
      }
    } else if (output_tensor->type == kTfLiteUInt8) {
      const uint8_t* out = reinterpret_cast<const uint8_t*>(output_tensor->data.raw);
      uint8_t max_val = out[0];
      for (int i = 1; i < output_elements; i++) {
        if (out[i] > max_val) { max_val = out[i]; top1 = i; }
      }
    }

    printf("%d\n", top1);
  }

  return 0;
}
