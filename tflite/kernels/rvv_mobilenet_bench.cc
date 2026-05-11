/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). */
// MobileNet inference benchmark for RVV verification.
//
// Single model:
//   ./rvv_mobilenet_bench <model.tflite> [num_runs] [num_threads]
//
// Multiple models (comparison table):
//   ./rvv_mobilenet_bench --all <dir> [num_runs]
//   Runs all .tflite files in <dir> with 1,4,8 threads.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

#include "tflite/kernels/internal/optimized/rvv_check.h"
#include "tflite/core/interpreter.h"
#include "tflite/core/interpreter_builder.h"
#include "tflite/core/model_builder.h"
#include "tflite/kernels/register.h"

struct Result {
  std::string model;
  int threads;
  double avg, p50, p95, min_l, max_l;
};

static const char* TypeName(TfLiteType t) {
  switch(t){case kTfLiteFloat32:return "fp32";case kTfLiteInt8:return "int8";
  case kTfLiteUInt8:return "uint8";default:return "other";}
}

Result BenchModel(const char* path, int num_runs, int num_threads) {
  Result r;
  r.model = path;
  r.threads = num_threads;
  r.avg = r.p50 = r.p95 = r.min_l = r.max_l = -1;

  auto model = tflite::FlatBufferModel::BuildFromFile(path);
  if (!model) { fprintf(stderr, "Failed to load: %s\n", path); return r; }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) { fprintf(stderr, "Failed to build interpreter\n"); return r; }

  interpreter->SetNumThreads(num_threads);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "Failed to allocate tensors\n"); return r;
  }

  // Warmup
  for (int i = 0; i < 5; i++) interpreter->Invoke();

  // Benchmark
  std::vector<double> lat(num_runs);
  for (int i = 0; i < num_runs; i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    interpreter->Invoke();
    auto t1 = std::chrono::high_resolution_clock::now();
    lat[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  std::sort(lat.begin(), lat.end());
  r.avg = std::accumulate(lat.begin(), lat.end(), 0.0) / num_runs;
  r.p50 = lat[num_runs / 2];
  r.p95 = lat[(int)(num_runs * 0.95)];
  r.min_l = lat.front();
  r.max_l = lat.back();
  return r;
}

void PrintSingle(const char* path, int num_runs, int num_threads) {
  auto model = tflite::FlatBufferModel::BuildFromFile(path);
  if (!model) { fprintf(stderr, "Failed to load: %s\n", path); return; }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder(*model, resolver)(&interp);
  if (!interp) return;
  interp->AllocateTensors();
  auto* inp = interp->input_tensor(0);

  printf("Model: %s\n", path);
  printf("Input: %s [", inp->name);
  for (int i = 0; i < inp->dims->size; i++)
    printf("%s%d", i ? "x" : "", inp->dims->data[i]);
  printf("] %s\n", TypeName(inp->type));
  printf("Threads: %d, Warmup: 5, Runs: %d\n", num_threads, num_runs);
#ifdef USE_RVV
  printf("USE_RVV: active\n\n");
#else
  printf("USE_RVV: not active (scalar fallback)\n\n");
#endif

  auto r = BenchModel(path, num_runs, num_threads);
  printf("Inference latency (ms):\n");
  printf("  avg:  %.2f\n", r.avg);
  printf("  p50:  %.2f\n", r.p50);
  printf("  p95:  %.2f\n", r.p95);
  printf("  min:  %.2f\n", r.min_l);
  printf("  max:  %.2f\n", r.max_l);
}

void RunAll(const char* dir, int num_runs) {
#ifdef USE_RVV
  printf("USE_RVV: active\n");
#else
  printf("USE_RVV: not active\n");
#endif
  printf("Directory: %s\n\n", dir);

  std::vector<std::string> models;
  DIR* d = opendir(dir);
  if (!d) { fprintf(stderr, "Cannot open dir: %s\n", dir); return; }
  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr) {
    std::string name(ent->d_name);
    if (name.size() > 7 && name.substr(name.size()-7) == ".tflite")
      models.push_back(std::string(dir) + "/" + name);
  }
  closedir(d);
  std::sort(models.begin(), models.end());

  printf("%-45s %4s %8s %8s %8s\n", "Model", "T", "avg(ms)", "p50(ms)", "p95(ms)");
  printf("%s\n", std::string(80, '-').c_str());

  int threads[] = {1, 4, 8};
  for (auto& m : models) {
    // Extract short name
    std::string short_name = m.substr(m.rfind('/') + 1);
    short_name = short_name.substr(0, short_name.size() - 7);  // remove .tflite

    for (int t : threads) {
      auto r = BenchModel(m.c_str(), num_runs, t);
      if (r.avg < 0) continue;
      printf("%-45s %4d %8.1f %8.1f %8.1f\n",
             short_name.c_str(), t, r.avg, r.p50, r.p95);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <model.tflite> [num_runs] [num_threads]\n", argv[0]);
    fprintf(stderr, "  %s --all <model_dir> [num_runs]\n", argv[0]);
    return 1;
  }

  if (strcmp(argv[1], "--all") == 0) {
    if (argc < 3) { fprintf(stderr, "Need model directory\n"); return 1; }
    int runs = argc > 3 ? atoi(argv[3]) : 10;
    RunAll(argv[2], runs);
  } else {
    int runs = argc > 2 ? atoi(argv[2]) : 20;
    int threads = argc > 3 ? atoi(argv[3]) : 1;
    PrintSingle(argv[1], runs, threads);
  }
  return 0;
}
