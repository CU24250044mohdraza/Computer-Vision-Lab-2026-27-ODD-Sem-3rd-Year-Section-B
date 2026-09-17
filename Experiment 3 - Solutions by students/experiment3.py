# Experiment No. 3
# Implementation and Analysis of Spatial Filtering Techniques using Low-Pass and High-Pass Filters

# ---------------------------------------------------------
# Step 1: Import the required libraries and load an image
# ---------------------------------------------------------
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

IMAGE_PATH = "input.jpeg"  # replace with your image path

img = cv2.imread(IMAGE_PATH)

if img is None:
    # Fallback: generate a sample image if no file is found
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (180, 150), (0, 0, 255), -1)
    cv2.rectangle(img, (200, 50), (350, 200), (0, 255, 0), -1)
    cv2.circle(img, (200, 250), 60, (255, 0, 0), -1)
    print(f"No image found at '{IMAGE_PATH}'. Generated a sample image instead.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add synthetic salt-and-pepper noise so median filtering has something to remove
noisy = gray.copy()
rng = np.random.default_rng(seed=42)
num_salt = int(0.02 * gray.size)
coords = (rng.integers(0, gray.shape[0], num_salt), rng.integers(0, gray.shape[1], num_salt))
noisy[coords] = 255
coords = (rng.integers(0, gray.shape[0], num_salt), rng.integers(0, gray.shape[1], num_salt))
noisy[coords] = 0

# ---------------------------------------------------------
# Step 2: Display the original image and analyze visual characteristics
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Original Grayscale"); axes[0].axis("off")
axes[1].imshow(noisy, cmap="gray"); axes[1].set_title("With Salt-and-Pepper Noise"); axes[1].axis("off")
plt.tight_layout()
plt.show()

print("Original Image Characteristics")
print("-------------------------------")
print(f"Shape: {gray.shape}, mean: {gray.mean():.2f}, std: {gray.std():.2f}")

# ---------------------------------------------------------
# Step 3: Gaussian Blur to reduce noise while preserving structure
# ---------------------------------------------------------
t0 = time.perf_counter()
gaussian_blur = cv2.GaussianBlur(noisy, (5, 5), 0)
t_gaussian = time.perf_counter() - t0

# ---------------------------------------------------------
# Step 4: Median Filtering to remove salt-and-pepper noise
# ---------------------------------------------------------
t0 = time.perf_counter()
median_blur = cv2.medianBlur(noisy, 5)
t_median = time.perf_counter() - t0

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(noisy, cmap="gray"); axes[0].set_title("Noisy Input"); axes[0].axis("off")
axes[1].imshow(gaussian_blur, cmap="gray"); axes[1].set_title("Gaussian Blur"); axes[1].axis("off")
axes[2].imshow(median_blur, cmap="gray"); axes[2].set_title("Median Filter"); axes[2].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 5: Average (Mean) Filtering and comparison with Gaussian
# ---------------------------------------------------------
t0 = time.perf_counter()
average_blur = cv2.blur(noisy, (5, 5))
t_average = time.perf_counter() - t0

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(average_blur, cmap="gray"); axes[0].set_title("Average (Mean) Filter"); axes[0].axis("off")
axes[1].imshow(gaussian_blur, cmap="gray"); axes[1].set_title("Gaussian Filter"); axes[1].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 6: Laplacian Filtering to enhance edges and fine details
# ---------------------------------------------------------
t0 = time.perf_counter()
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplacian_abs = cv2.convertScaleAbs(laplacian)
t_laplacian = time.perf_counter() - t0

plt.figure(figsize=(5, 4))
plt.imshow(laplacian_abs, cmap="gray")
plt.title("Laplacian Filter (Edges)")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 7: Sobel Edge Detection (horizontal and vertical)
# ---------------------------------------------------------
t0 = time.perf_counter()
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
t_sobel = time.perf_counter() - t0

sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
sobel_combined_abs = cv2.convertScaleAbs(sobel_combined)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(sobel_x_abs, cmap="gray"); axes[0].set_title("Sobel X (Vertical Edges)"); axes[0].axis("off")
axes[1].imshow(sobel_y_abs, cmap="gray"); axes[1].set_title("Sobel Y (Horizontal Edges)"); axes[1].axis("off")
axes[2].imshow(sobel_combined_abs, cmap="gray"); axes[2].set_title("Sobel Combined"); axes[2].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 8: Compare filters - noise reduction, edge preservation, performance
# ---------------------------------------------------------
def noise_reduction_score(filtered):
    # Lower std relative to noisy image suggests more smoothing/noise reduction
    return filtered.std()

print("\nLow-Pass Filter Comparison")
print("---------------------------")
print(f"{'Filter':20s} | {'Std (lower~smoother)':22s} | {'Time (s)':10s}")
print(f"{'Noisy Input':20s} | {noisy.std():22.2f} | {'-':>10s}")
print(f"{'Average (Mean)':20s} | {noise_reduction_score(average_blur):22.2f} | {t_average:10.6f}")
print(f"{'Gaussian':20s} | {noise_reduction_score(gaussian_blur):22.2f} | {t_gaussian:10.6f}")
print(f"{'Median':20s} | {noise_reduction_score(median_blur):22.2f} | {t_median:10.6f}")

print("\nHigh-Pass Filter Comparison")
print("----------------------------")
print(f"{'Filter':20s} | {'Edge Response (mean)':22s} | {'Time (s)':10s}")
print(f"{'Laplacian':20s} | {laplacian_abs.mean():22.2f} | {t_laplacian:10.6f}")
print(f"{'Sobel Combined':20s} | {sobel_combined_abs.mean():22.2f} | {t_sobel:10.6f}")

# ---------------------------------------------------------
# Step 9: Display original and filtered images together with observations
# ---------------------------------------------------------
titles = ["Original", "Noisy", "Average", "Gaussian", "Median", "Laplacian", "Sobel X", "Sobel Y"]
images = [gray, noisy, average_blur, gaussian_blur, median_blur,
          laplacian_abs, sobel_x_abs, sobel_y_abs]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, title, image in zip(axes.ravel(), titles, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 10: Interpret results and discuss applications
# ---------------------------------------------------------
print("\nObservations")
print("------------")
print("- Average filtering smooths noise but blurs edges uniformly, losing fine detail.")
print("- Gaussian filtering smooths similarly to averaging but weights nearby pixels more, "
      "giving a more natural blur with better edge preservation than the average filter.")
print("- Median filtering is most effective against salt-and-pepper noise since it replaces "
      "each pixel with the median of its neighborhood, discarding extreme outlier values "
      "rather than averaging them in.")
print("- Laplacian filtering highlights regions of rapid intensity change (edges, fine detail) "
      "but is sensitive to noise since it is a second-derivative operator.")
print("- Sobel filtering detects directional gradients (horizontal/vertical) and is more "
      "robust to noise than Laplacian due to its built-in smoothing.")
print("- There is a fundamental trade-off between smoothing (noise reduction) and edge "
      "preservation: low-pass filters reduce noise at the cost of detail, while high-pass "
      "filters enhance detail but can amplify noise.")
print("- Applications: medical image denoising, edge-based object detection, satellite image "
      "preprocessing, and autonomous vehicle perception pipelines.")
