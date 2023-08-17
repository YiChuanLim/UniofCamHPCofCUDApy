#Example 1: Vector Addition in traditional python
import numpy as np
import time

#Create two large vectors

n = 10000000
a = np.random.rand(n)
b = np.random.rand(n)

# Calculate vector addition
start_time = time.time()
result = a + b
end_time = time.time()

print("Time taken (Traditional Python):", end_time - start_time, "seconds")
