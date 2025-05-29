import torch
import time

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Create sample tensors and perform a matrix multiplication
size = 1000
a = torch.randn(size, size).to(device)
b = torch.randn(size, size).to(device)

# Measure time on GPU
start_time = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # Ensure GPU computation is complete
gpu_time = time.time() - start_time
print(f"Matrix multiplication on {device}: {gpu_time:.4f} seconds")

# Compare with CPU
a_cpu = a.to('cpu')
b_cpu = b.to('cpu')
start_time = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start_time
print(f"Matrix multiplication on CPU: {cpu_time:.4f} seconds")