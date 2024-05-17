import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import main

def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

original_image = np.array(Image.open("images/snail.bmp"))

compressed_data_numpy = main.compress_image_svd(original_image, 'numpy', compression=10)
compressed_data_simple = main.compress_image_svd(original_image, 'simple', compression=10)
compressed_data_advanced = main.compress_image_svd(original_image, 'advanced', compression=10)

main.save_compressed_data("compressed_numpy.pkl", compressed_data_numpy)
main.save_compressed_data("compressed_simple.pkl", compressed_data_simple)
main.save_compressed_data("compressed_advanced.pkl", compressed_data_advanced)

decompressed_image_numpy = main.decompress_image_svd(compressed_data_numpy)
decompressed_image_simple = main.decompress_image_svd(compressed_data_simple)
decompressed_image_advanced = main.decompress_image_svd(compressed_data_advanced)

Image.fromarray(decompressed_image_numpy).save("decompressed_numpy.bmp")
Image.fromarray(decompressed_image_simple).save("decompressed_simple.bmp")
Image.fromarray(decompressed_image_advanced).save("decompressed_advanced.bmp")

mse_numpy = calculate_mse(original_image, decompressed_image_numpy)
mse_simple = calculate_mse(original_image, decompressed_image_simple)
mse_advanced = calculate_mse(original_image, decompressed_image_advanced)

print(f"MSE for Numpy method: {mse_numpy}")
print(f"MSE for Simple method: {mse_simple}")
print(f"MSE for Advanced method: {mse_advanced}")

# Plot the original and decompressed images for visual comparison
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[1].imshow(decompressed_image_numpy)
axes[1].set_title("Numpy Decompressed")
axes[2].imshow(decompressed_image_simple)
axes[2].set_title("Simple Decompressed")
axes[3].imshow(decompressed_image_advanced)
axes[3].set_title("Advanced Decompressed")

for ax in axes:
    ax.axis('off')

plt.show()
