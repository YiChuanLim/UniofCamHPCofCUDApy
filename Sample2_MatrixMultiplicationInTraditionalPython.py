#Example 2: Matrix Multiplication in Traditional Python
import numpy as np
import time

# Create two large matrices
n = 1000
a = np.random.rand(n, n)
b = np.random.rand(n, n)

# Calculate matrix multiplication
start_time = time.time()
result = np.dot(a, b)
end_time = time.time()

print("Time taken (Traditional Python):", end_time - start_time, "seconds")
