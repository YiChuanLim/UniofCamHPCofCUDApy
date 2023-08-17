import numpy as np
import time

# create a large image
image_size = (1920, 1080)
image = np.random.randint(0, 256, image_size, dtype=np.uint8)

# Analog filter operation
def apply_filter(image):
    filtered_image = np.empty_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # filter operation
            filtered_image[i, j] = image[i, j] * 0.5  # 简化的滤波操作
    return filtered_image

start_time = time.time()
filtered_image = apply_filter(image)
end_time = time.time()

print("Time taken (Traditional Python):", end_time - start_time, "seconds")
