import numpy as np
from numba import cuda
import time

# create a large image
image_size = (1920, 1080)
image = np.random.randint(0, 256, image_size, dtype=np.uint8)
filtered_image = np.empty_like(image)

#Image Filtering Accelerated Using CUDA@cuda.jit
def apply_filter_kernel(image, filtered_image):
    i, j = cuda.grid(2)
    if i < image.shape[0] and j < image.shape[1]:
        # filter operation
        filtered_image[i, j] = image[i, j] * 0.5  # Simplified filtering operations

threads_per_block = (16, 16)
blocks_per_grid = ((image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                   (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])

start_time = time.time()
apply_filter_kernel[blocks_per_grid, threads_per_block](image, filtered_image)
cuda.synchronize()
end_time = time.time()

print("Time taken (CUDA Python):", end_time - start_time, "seconds")
