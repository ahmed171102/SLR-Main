"""
GPU Verification Script
Run this script to verify TensorFlow is using NVIDIA GPU
"""
import tensorflow as tf
import subprocess
import sys

print("=" * 60)
print("GPU VERIFICATION FOR TENSORFLOW")
print("=" * 60)

# 1. Check TensorFlow GPU detection
print("\n1. TENSORFLOW GPU DETECTION:")
print("-" * 60)
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPUs Detected by TensorFlow: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu.name}")
    try:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"    Details: {details}")
    except:
        pass

# 2. Check which device TensorFlow will use
print("\n2. DEFAULT DEVICE:")
print("-" * 60)
print(f"Default device: {tf.config.list_physical_devices('GPU')[0].name if gpus else 'CPU'}")

# 3. Run a test computation and check device
print("\n3. TEST COMPUTATION:")
print("-" * 60)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(f"Test computation executed on: {c.device}")
    print(f"Result shape: {c.shape}")

# 4. Check NVIDIA GPU via nvidia-smi
print("\n4. NVIDIA GPU INFORMATION (via nvidia-smi):")
print("-" * 60)
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', 
                            '--format=csv,noheader'], 
                           capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("NVIDIA GPU Status:")
        print(result.stdout)
    else:
        print("nvidia-smi not available or error occurred")
except FileNotFoundError:
    print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
except Exception as e:
    print(f"Error running nvidia-smi: {e}")

# 5. Check CUDA availability
print("\n5. CUDA INFORMATION:")
print("-" * 60)
print(f"CUDA Built: {tf.test.is_built_with_cuda()}")
if gpus:
    print(f"CUDA Available: True")
    try:
        # Try to get CUDA version
        import tensorflow as tf
        print("GPU Device Name:", gpus[0].name)
    except:
        pass
else:
    print("CUDA Available: False")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nTo monitor GPU usage during training, run:")
print("  nvidia-smi -l 1")
print("(This will refresh every 1 second showing GPU utilization)")

