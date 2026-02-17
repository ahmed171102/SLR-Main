# Training Time & GPU Memory Usage Explanation

## 1. EPOCH TIME CALCULATION

### Your Current Setup:
- **Training samples**: 69,600 images
- **Batch size**: 32
- **Image size**: 128x128x3
- **Model**: MobileNetV2 (base + custom layers)
- **Steps per epoch**: 69,600 ÷ 32 = **2,175 steps**

### Time Calculation:

**Formula:**
```
Epoch Time = (Steps per Epoch) × (Time per Step)
```

**Factors affecting time per step:**
1. **Model complexity** (MobileNetV2 forward + backward pass)
2. **Data loading** (reading images from disk)
3. **Data augmentation** (rotation, shift, zoom, etc.)
4. **GPU speed** (NVIDIA GeForce MX150 is entry-level)
5. **CPU speed** (data preprocessing bottleneck)

### Estimated Time per Step:
- **Fast GPU (RTX 3080)**: ~0.1-0.2 seconds/step → **4-7 minutes/epoch**
- **Mid-range GPU (GTX 1660)**: ~0.3-0.5 seconds/step → **11-18 minutes/epoch**
- **Your GPU (MX150)**: ~1.5-2.0 seconds/step → **54-73 minutes/epoch** ✓

### Why 1 Hour is Normal for Your Setup:

1. **MX150 is entry-level**: 
   - 384 CUDA cores (vs 2944 in RTX 3080)
   - Lower memory bandwidth
   - Designed for light workloads

2. **Large dataset**: 69,600 images is substantial

3. **Data augmentation**: Each image is transformed (rotation, shift, etc.) which adds CPU overhead

4. **Data loading bottleneck**: 
   - Reading 32 images from disk per batch
   - Preprocessing on CPU while GPU waits
   - This causes the 0% GPU utilization you see between batches

### How to Speed Up Training:

1. **Increase batch size** (if memory allows):
   - Current: 32 → Try: 64 or 128
   - Reduces steps per epoch: 2,175 → 1,088 (batch 64)
   - **Time savings: ~30-40%**

2. **Reduce data augmentation** (temporarily for speed):
   - Remove some augmentations during initial training
   - Add them back for fine-tuning

3. **Use prefetching** (already optimized in your code)

4. **Reduce image size** (if accuracy allows):
   - Current: 128x128 → Try: 96x96
   - **Time savings: ~20-30%**

5. **Use mixed precision** (already enabled in your code)

---

## 2. GPU MEMORY USAGE (Why 2GB instead of 4GB?)

### Your Current Usage:
- **Total GPU Memory**: 4,096 MiB (4 GB)
- **Used**: 2,371 MiB (~2.3 GB)
- **Available**: 1,725 MiB (~1.7 GB)
- **Usage**: ~58%

### Why TensorFlow Doesn't Use All Memory:

#### 1. **Memory Growth Policy** (You enabled this!)
```python
tf.config.experimental.set_memory_growth(gpu, True)
```
- TensorFlow allocates memory **on-demand** (as needed)
- Doesn't pre-allocate all 4GB
- **This is GOOD** - prevents memory fragmentation and allows other programs to use GPU

#### 2. **Actual Memory Requirements:**

**Memory Breakdown:**
- **Model weights**: ~50-100 MB (MobileNetV2 base)
- **Model activations**: ~200-400 MB (during forward pass)
- **Gradients**: ~50-100 MB (during backward pass)
- **Optimizer states** (Adam): ~100-200 MB (momentum, variance)
- **Batch data**: ~32 images × 128×128×3 × 4 bytes = ~6 MB
- **Intermediate tensors**: ~500-1000 MB
- **TensorFlow overhead**: ~200-500 MB
- **CUDA/cuDNN libraries**: ~200-300 MB

**Total**: ~1,300-2,500 MB (matches your 2,371 MB!)

#### 3. **Why Not Use All 4GB?**

**Reasons:**
1. **Not needed**: Your model doesn't require more memory
2. **Safety margin**: Prevents out-of-memory errors
3. **Other processes**: Windows, display driver, etc. need GPU memory
4. **Memory fragmentation**: Leaving space prevents issues

### How to Use More Memory (If Needed):

#### Option 1: Increase Batch Size
```python
BATCH_SIZE = 64  # or 128
```
- **Memory increase**: ~2x for batch 64, ~4x for batch 128
- **Benefit**: Faster training (fewer steps)
- **Risk**: Out of memory if too large

#### Option 2: Disable Memory Growth (Not Recommended)
```python
# DON'T do this unless you have a specific reason
# tf.config.experimental.set_memory_growth(gpu, False)
```

#### Option 3: Use Larger Images
- 128×128 → 224×224 (MobileNetV2 default)
- **Memory increase**: ~3x
- **Benefit**: Better accuracy potentially
- **Risk**: May exceed memory

### Current Memory Usage is OPTIMAL:

✅ **2.3 GB / 4 GB (58%)** is perfect because:
- Leaves room for batch size increases
- Prevents memory errors
- Allows other applications to use GPU
- Efficient memory utilization

---

## SUMMARY

### Epoch Time:
- **1 hour per epoch is NORMAL** for your setup (MX150 + 69,600 images)
- **Calculation**: 2,175 steps × ~1.7 seconds/step = ~1 hour
- **To speed up**: Increase batch size to 64 or 128 (if memory allows)

### GPU Memory:
- **2.3 GB usage is CORRECT** - TensorFlow only uses what it needs
- **Not using 4 GB is GOOD** - prevents memory issues and allows flexibility
- **To use more**: Increase batch size (will automatically use more memory)

### Recommendations:
1. ✅ Current setup is working correctly
2. ⚠️ Try increasing batch size to 64 (will use ~3-3.5 GB, still safe)
3. ⚠️ Monitor for out-of-memory errors if you increase batch size
4. ✅ 1 hour/epoch is expected for your hardware

