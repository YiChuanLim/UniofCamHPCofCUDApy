#Example 1: Vector Addition in CUDA Python
import numpy as np
from numba import cuda
import time

# Create two large vectors
n = 10000000
a = np.random.rand(n)
b = np.random.rand(n)
result = np.empty_like(a)

# Vector addition accelerated using CUDA
@cuda.jit
def add_kernel(a, b, result):
    idx = cuda.grid(1)
    if idx < result.shape[0]:
        result[idx] = a[idx] + b[idx]

threads_per_block = 128
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

start_time = time.time()
add_kernel[blocks_per_grid, threads_per_block](a, b, result)
cuda.synchronize()
end_time = time.time()

print("Time taken (CUDA Python):", end_time - start_time, "seconds")
