# Experiment No. 6
# Implementation and Comparative Analysis of Image Segmentation Techniques using Python and OpenCV

# ---------------------------------------------------------
# Step 1: Import the required libraries and load an image
# ---------------------------------------------------------
# pip install opencv-python numpy matplotlib scikit-learn

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------
# Step 2: Convert to grayscale and reduce noise with Gaussian Blur
# ---------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------------------------------------------------
# Step 3: Global Thresholding
# ---------------------------------------------------------
t0 = time.perf_counter()
_, global_thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
t_global = time.perf_counter() - t0

# ---------------------------------------------------------
# Step 4: Otsu's Thresholding
# ---------------------------------------------------------
t0 = time.perf_counter()
otsu_val, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
t_otsu = time.perf_counter() - t0

print("Thresholding Comparison")
print("-----------------------")
print(f"Global threshold (fixed): 127")
print(f"Otsu's optimal threshold (auto): {otsu_val:.1f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Grayscale"); axes[0].axis("off")
axes[1].imshow(global_thresh, cmap="gray"); axes[1].set_title("Global Threshold (127)"); axes[1].axis("off")
axes[2].imshow(otsu_thresh, cmap="gray"); axes[2].set_title(f"Otsu Threshold ({otsu_val:.0f})"); axes[2].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 5: Adaptive Thresholding (uneven illumination)
# ---------------------------------------------------------
t0 = time.perf_counter()
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
t_adaptive = time.perf_counter() - t0

plt.figure(figsize=(5, 4))
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 6: Watershed Segmentation (separate overlapping/touching objects)
# ---------------------------------------------------------
t0 = time.perf_counter()

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

watershed_img = img.copy()
markers = cv2.watershed(watershed_img, markers)
watershed_img[markers == -1] = [0, 0, 255]  # mark boundaries in red

t_watershed = time.perf_counter() - t0

num_regions = len(np.unique(markers)) - 2  # exclude background(1) and boundary(-1)
print(f"\nWatershed Segmentation")
print("-----------------------")
print(f"Number of segmented regions detected: {max(num_regions, 0)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB)); axes[1].set_title("Watershed Boundaries")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 7: K-Means Clustering (color-based segmentation)
# ---------------------------------------------------------
t0 = time.perf_counter()

pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)
segmented_kmeans = centers[labels].reshape(img_rgb.shape)

t_kmeans = time.perf_counter() - t0

plt.figure(figsize=(5, 4))
plt.imshow(segmented_kmeans)
plt.title(f"K-Means Segmentation (K={K})")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 8: Compare segmentation outputs (boundaries, accuracy, efficiency)
# ---------------------------------------------------------
print("\nComputational Performance Comparison")
print("-------------------------------------")
print(f"{'Technique':25s} | {'Time (s)':10s}")
print(f"{'Global Thresholding':25s} | {t_global:10.6f}")
print(f"{'Otsu Thresholding':25s} | {t_otsu:10.6f}")
print(f"{'Adaptive Thresholding':25s} | {t_adaptive:10.6f}")
print(f"{'Watershed':25s} | {t_watershed:10.6f}")
print(f"{'K-Means (K={})'.format(K):25s} | {t_kmeans:10.6f}")

# ---------------------------------------------------------
# Step 9: Visualize all segmented images together
# ---------------------------------------------------------
titles = ["Original", "Global Threshold", "Otsu Threshold", "Adaptive Threshold",
          "Watershed", "K-Means"]
images = [img_rgb, global_thresh, otsu_thresh, adaptive_thresh,
          cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB), segmented_kmeans]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, title, image in zip(axes.ravel(), titles, images):
    cmap = "gray" if image.ndim == 2 else None
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 10: Summarize observations
# ---------------------------------------------------------
print("\nObservations")
print("------------")
print("- Global Thresholding uses one fixed value (127) for the whole image, which works "
      "only when illumination is uniform; it may over- or under-segment otherwise.")
print("- Otsu's Thresholding automatically computes the optimal threshold from the image "
      "histogram (found here: {:.1f}), removing the need to manually tune a fixed value.".format(otsu_val))
print("- Adaptive Thresholding computes a local threshold for each neighborhood, making it "
      "much more robust on images with uneven illumination or shadows.")
print("- Watershed Segmentation treats the image as a topographic surface and floods it "
      "from markers, effectively separating touching/overlapping objects that thresholding "
      "alone cannot split.")
print("- K-Means Clustering groups pixels by color similarity rather than intensity alone, "
      "producing smooth region-based segmentation useful for scenes with multiple distinct "
      "colors, at higher computational cost than simple thresholding.")
print("- Threshold-based methods are fast and simple but limited to intensity-based "
      "separation; clustering/marker-based methods (K-Means, Watershed) handle more complex "
      "scenes but cost more compute.")
print("- Applications: medical image analysis (tumor/organ segmentation), satellite land-use "
      "mapping, autonomous driving (road/obstacle segmentation), industrial defect detection, "
      "and document/text segmentation.")
