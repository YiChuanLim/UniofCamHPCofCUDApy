import numpy as np
from numba import cuda
import time

# 创建两个大型矩阵
n = 1000
a = np.random.rand(n, n)
b = np.random.rand(n, n)
result = np.empty((n, n), dtype=np.float64)

# 使用CUDA加速的矩阵乘法
@cuda.jit
def matrix_mult_kernel(a, b, result):
    i, j = cuda.grid(2)
    if i < n and j < n:
        temp = 0
        for k in range(n):
            temp += a[i, k] * b[k, j]
        result[i, j] = temp

threads_per_block = (16, 16)
blocks_per_grid = ((n + threads_per_block[0] - 1) // threads_per_block[0],
                   (n + threads_per_block[1] - 1) // threads_per_block[1])

start_time = time.time()
matrix_mult_kernel[blocks_per_grid, threads_per_block](a, b, result)
cuda.synchronize()
end_time = time.time()

print("Time taken (CUDA Python):", end_time - start_time, "seconds")
