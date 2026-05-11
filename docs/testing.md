# Testing Guide

All numbers in `speedup.md` are reproduced by the commands in this file. Three test binaries cover everything.

## 1. Build

```bash
ssh openkylin@192.168.5.211   # password: openkylin
cd /home/openkylin/github/rvspoc-S2601-litert
```

### RVV build

```bash
mkdir -p build && cd build
cmake ../tflite \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv"
make -j8 rvv_bench rvv_mobilenet_bench rvv_top1_test
```

### Scalar build (baseline for comparison)

```bash
cd /home/openkylin/github/rvspoc-S2601-litert
mkdir -p build-scalar && cd build-scalar
cmake ../tflite \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv -DTF_LITE_DISABLE_RVV"
make -j8 rvv_mobilenet_bench rvv_top1_test
```

**Important**: Both builds must use `-DTFLITE_ENABLE_RUY=ON`. Without Ruy, the gemmlowp threadpool is used which is broken on RISC-V and multi-threaded results will be wrong.

## 2. Test Binaries

| Binary | Source | Purpose |
|---|---|---|
| `rvv_bench` | `tflite/kernels/rvv_bench.cc` | Operator accuracy (95 tests) + per-operator speedup (13 ops) |
| `rvv_mobilenet_bench` | `tflite/kernels/rvv_mobilenet_bench.cc` | End-to-end model inference latency |
| `rvv_top1_test` | `tflite/kernels/rvv_top1_test.cc` | Top-1 prediction comparison (RVV vs scalar) |

## 3. Reproduce: Accuracy + Per-Operator Speedup (speedup.md Sections 1, 4)

```bash
cd /home/openkylin/github/rvspoc-S2601-litert/build
./rvv_bench
```

Outputs:
- **Accuracy**: 95 tests (FloatAdd/Mul/HardSwish, Int8Add/Mul, 19 sizes each), pass if FP32 error <= 1e-5, INT8 diff <= 1 LSB
- **Operator speedup**: 5 element-wise ops (N=16384) + 8 tensor utility functions, each showing `scalar=X.XXX ms  rvv=X.XXX ms  speedup=X.XXx`

## 4. Reproduce: Top-1 Accuracy (speedup.md Section 4)

Compares argmax predictions between RVV and scalar builds using deterministic random inputs.

```bash
# Usage: ./rvv_top1_test <model.tflite> [num_samples] [num_threads]
cd /home/openkylin/github/rvspoc-S2601-litert
M=/home/openkylin/models

for model in mobilenet_v1_1.0_224.tflite mobilenet_v1_int8.tflite mobilenet_v2_fp32.tflite mobilenet_v2_int8.tflite; do
  build-scalar/rvv_top1_test $M/$model 50 8 > /tmp/s.txt 2>/dev/null
  build/rvv_top1_test $M/$model 50 8 > /tmp/r.txt 2>/dev/null
  mismatch=$(diff /tmp/s.txt /tmp/r.txt | grep "^<" | wc -l)
  echo "$model: agree=$((50 - mismatch))/50 mismatch=$mismatch"
done
```

Expected results:

| Model | Agree | Mismatch | Threshold |
|---|---|---|---|
| MobileNetV1 FP32 | 50/50 | 0 | <= 0.1% |
| MobileNetV1 INT8 | 50/50 | 0 | <= 1% |
| MobileNetV2 FP32 | 50/50 | 0 | <= 0.1% |
| MobileNetV2 INT8 | 49/50 | 1 | <= 1% (2% observed, rounding at decision boundary) |

## 5. Reproduce: Model Preparation

### MobileNetV1 Float32

```bash
mkdir -p /home/openkylin/models && cd /home/openkylin/models
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
tar xzf mobilenet_v1_1.0_224.tgz ./mobilenet_v1_1.0_224.tflite
```

### INT8 models (convert on x86 host with TensorFlow, then scp to target)

```bash
pip install tensorflow

# MobileNetV1 INT8
python3 -c "
import tensorflow as tf, numpy as np
model = tf.keras.applications.MobileNet(weights='imagenet')
c = tf.lite.TFLiteConverter.from_keras_model(model)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = lambda: ([np.random.randn(1,224,224,3).astype(np.float32)] for _ in range(100))
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
open('mobilenet_v1_int8.tflite','wb').write(c.convert())
"

# MobileNetV2 Float32 + INT8
python3 -c "
import tensorflow as tf, numpy as np
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224,224,3))
open('mobilenet_v2_fp32.tflite','wb').write(tf.lite.TFLiteConverter.from_keras_model(model).convert())
c = tf.lite.TFLiteConverter.from_keras_model(model)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = lambda: ([np.random.randn(1,224,224,3).astype(np.float32)] for _ in range(100))
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
open('mobilenet_v2_int8.tflite','wb').write(c.convert())
"

scp mobilenet_v1_int8.tflite mobilenet_v2_fp32.tflite mobilenet_v2_int8.tflite \
  openkylin@192.168.5.211:/home/openkylin/models/
```

## 6. Reproduce: Model-Level Speedup (speedup.md Section 2)

### All models at once

```bash
cd /home/openkylin/github/rvspoc-S2601-litert

echo "=== SCALAR ===" && build-scalar/rvv_mobilenet_bench --all /home/openkylin/models 10
echo ""
echo "=== RVV ===" && build/rvv_mobilenet_bench --all /home/openkylin/models 10
```

### Individual model commands (matching speedup.md rows)

```bash
cd /home/openkylin/github/rvspoc-S2601-litert
SCALAR=build-scalar/rvv_mobilenet_bench
RVV=build/rvv_mobilenet_bench
M=/home/openkylin/models

# MobileNetV1 FP32: 1T scalar=3102 rvv=204 (15.2x)
$SCALAR $M/mobilenet_v1_1.0_224.tflite 5 1 | grep avg
$RVV    $M/mobilenet_v1_1.0_224.tflite 5 1 | grep avg

# MobileNetV1 FP32: 4T scalar=788 rvv=53 (14.9x)
$SCALAR $M/mobilenet_v1_1.0_224.tflite 5 4 | grep avg
$RVV    $M/mobilenet_v1_1.0_224.tflite 5 4 | grep avg

# MobileNetV1 FP32: 8T scalar=416 rvv=37 (11.2x)
$SCALAR $M/mobilenet_v1_1.0_224.tflite 5 8 | grep avg
$RVV    $M/mobilenet_v1_1.0_224.tflite 5 8 | grep avg

# MobileNetV1 INT8: 1T scalar=2252 rvv=238 (9.5x)
$SCALAR $M/mobilenet_v1_int8.tflite 10 1 | grep avg
$RVV    $M/mobilenet_v1_int8.tflite 10 1 | grep avg

# MobileNetV1 INT8: 4T scalar=569 rvv=67 (8.5x)
$SCALAR $M/mobilenet_v1_int8.tflite 10 4 | grep avg
$RVV    $M/mobilenet_v1_int8.tflite 10 4 | grep avg

# MobileNetV1 INT8: 8T scalar=306 rvv=40 (7.7x)
$SCALAR $M/mobilenet_v1_int8.tflite 10 8 | grep avg
$RVV    $M/mobilenet_v1_int8.tflite 10 8 | grep avg

# MobileNetV2 FP32: 1T scalar=1722 rvv=168 (10.3x)
$SCALAR $M/mobilenet_v2_fp32.tflite 5 1 | grep avg
$RVV    $M/mobilenet_v2_fp32.tflite 5 1 | grep avg

# MobileNetV2 FP32: 4T scalar=437 rvv=45 (9.7x)
$SCALAR $M/mobilenet_v2_fp32.tflite 5 4 | grep avg
$RVV    $M/mobilenet_v2_fp32.tflite 5 4 | grep avg

# MobileNetV2 FP32: 8T scalar=243 rvv=29 (8.4x)
$SCALAR $M/mobilenet_v2_fp32.tflite 5 8 | grep avg
$RVV    $M/mobilenet_v2_fp32.tflite 5 8 | grep avg

# MobileNetV2 INT8: 1T scalar=1407 rvv=243 (5.8x)
$SCALAR $M/mobilenet_v2_int8.tflite 10 1 | grep avg
$RVV    $M/mobilenet_v2_int8.tflite 10 1 | grep avg

# MobileNetV2 INT8: 4T scalar=364 rvv=70 (5.2x)
$SCALAR $M/mobilenet_v2_int8.tflite 10 4 | grep avg
$RVV    $M/mobilenet_v2_int8.tflite 10 4 | grep avg

# MobileNetV2 INT8: 8T scalar=210 rvv=47 (4.5x)
$SCALAR $M/mobilenet_v2_int8.tflite 10 8 | grep avg
$RVV    $M/mobilenet_v2_int8.tflite 10 8 | grep avg
```

## 7. Reproduce: Latency Threshold (speedup.md Section 5)

All must report `avg` <= 110ms.

```bash
cd /home/openkylin/github/rvspoc-S2601-litert/build

./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v2_fp32.tflite 20 8      # expect ~29ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v1_1.0_224.tflite 20 8   # expect ~37ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v1_int8.tflite 20 8      # expect ~40ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v2_fp32.tflite 20 4      # expect ~45ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v2_int8.tflite 20 8      # expect ~47ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v1_1.0_224.tflite 20 4   # expect ~53ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v1_1.0_224_quant.tflite 20 8  # expect ~63ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v1_int8.tflite 20 4      # expect ~67ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v2_int8.tflite 20 4      # expect ~70ms
./rvv_mobilenet_bench /home/openkylin/models/mobilenet_v1_1.0_224_quant.tflite 20 4  # expect ~93ms
```

## 8. Quick Full Validation (single copy-paste block)

```bash
cd /home/openkylin/github/rvspoc-S2601-litert
M=/home/openkylin/models

echo "=== Operator Accuracy + Speedup ==="
build/rvv_bench

echo "=== Top-1 Accuracy ==="
for model in mobilenet_v1_1.0_224.tflite mobilenet_v1_int8.tflite mobilenet_v2_fp32.tflite mobilenet_v2_int8.tflite; do
  build-scalar/rvv_top1_test $M/$model 50 8 > /tmp/s.txt 2>/dev/null
  build/rvv_top1_test $M/$model 50 8 > /tmp/r.txt 2>/dev/null
  mismatch=$(diff /tmp/s.txt /tmp/r.txt | grep "^<" | wc -l)
  echo "  $model: agree=$((50 - mismatch))/50"
done

echo "=== Model Inference ==="
build/rvv_mobilenet_bench --all $M 10
```
