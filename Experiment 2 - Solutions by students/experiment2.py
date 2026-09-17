# Experiment No. 2
# Contrast Enhancement and Histogram-Based Image Processing using Python and OpenCV

# ---------------------------------------------------------
# Step 1: Import the required libraries
# ---------------------------------------------------------
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "input.jpeg"  # replace with your image path

# ---------------------------------------------------------
# Step 2: Load a low-contrast grayscale image (or convert color -> grayscale)
# ---------------------------------------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    # Fallback: generate a synthetic low-contrast grayscale image
    base = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(base, (30, 30), (180, 150), 120, -1)
    cv2.rectangle(base, (200, 50), (350, 200), 140, -1)
    cv2.circle(base, (200, 250), 60, 100, -1)
    noise = np.random.randint(-5, 5, base.shape, dtype=np.int16)
    gray = np.clip(base.astype(np.int16) + noise + 100, 0, 255).astype(np.uint8)
    print(f"No image found at '{IMAGE_PATH}'. Generated a synthetic low-contrast image instead.")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------
# Step 3: Display the original image and generate its histogram
# ---------------------------------------------------------
hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale Image")
axes[0].axis("off")
axes[1].plot(hist_original, color="black")
axes[1].set_title("Original Histogram")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 4: Contrast stretching to improve the dynamic range
# ---------------------------------------------------------
min_val, max_val = np.min(gray), np.max(gray)
stretched = ((gray.astype(np.float32) - min_val) * 255.0 / (max_val - min_val))
stretched = np.clip(stretched, 0, 255).astype(np.uint8)

print(f"Original intensity range: [{min_val}, {max_val}]")
print(f"Stretched intensity range: [{stretched.min()}, {stretched.max()}]")

plt.figure(figsize=(5, 4))
plt.imshow(stretched, cmap="gray")
plt.title("Contrast Stretched Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 5: Histogram Equalization and comparison with original
# ---------------------------------------------------------
equalized = cv2.equalizeHist(gray)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(equalized, cmap="gray")
axes[1].set_title("Histogram Equalized")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 6: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(equalized, cmap="gray"); axes[1].set_title("Histogram Equalized"); axes[1].axis("off")
axes[2].imshow(clahe_img, cmap="gray"); axes[2].set_title("CLAHE"); axes[2].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 7: Plot and compare histograms of original, equalized, and CLAHE images
# ---------------------------------------------------------
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
hist_clahe = cv2.calcHist([clahe_img], [0], None, [256], [0, 256])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(hist_original, color="black"); axes[0].set_title("Original Histogram")
axes[1].plot(hist_equalized, color="blue"); axes[1].set_title("Equalized Histogram")
axes[2].plot(hist_clahe, color="green"); axes[2].set_title("CLAHE Histogram")
for ax in axes:
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 8: Analyze visual improvements (brightness, contrast, feature visibility)
# ---------------------------------------------------------
def summarize(name, image):
    print(f"{name:20s} | mean: {image.mean():7.2f} | std (contrast): {image.std():7.2f} "
          f"| min: {image.min():3d} | max: {image.max():3d}")

print("\nQuantitative Comparison")
print("-----------------------")
summarize("Original", gray)
summarize("Contrast Stretched", stretched)
summarize("Histogram Equalized", equalized)
summarize("CLAHE", clahe_img)

# ---------------------------------------------------------
# Step 9: Compare effectiveness of Histogram Equalization vs CLAHE
# ---------------------------------------------------------
print("\nComparison: Histogram Equalization vs CLAHE")
print("--------------------------------------------")
print(f"HE std (global contrast gain): {equalized.std():.2f}")
print(f"CLAHE std (local contrast gain): {clahe_img.std():.2f}")
print("Histogram Equalization applies a single global mapping and can over-amplify noise "
      "or wash out regions with very different local lighting.")
print("CLAHE applies equalization on local tiles with a clip limit, giving more balanced "
      "enhancement across regions with varying illumination, at the cost of extra computation.")

# ---------------------------------------------------------
# Step 10: Document observations
# ---------------------------------------------------------
print("\nObservations")
print("------------")
print("- The original histogram shows pixel intensities concentrated in a narrow range, "
      "indicating low contrast.")
print("- Contrast stretching linearly expands this range to [0, 255], improving contrast "
      "without changing the pixel distribution shape.")
print("- Histogram Equalization redistributes intensities to flatten the histogram, "
      "producing a strong global contrast boost but potentially over-enhancing noise.")
print("- CLAHE enhances contrast locally (per tile) with a clip limit to avoid over-amplification, "
      "performing better on images with non-uniform illumination.")
print("- CLAHE is generally preferred for medical and low-light imagery where local detail "
      "matters more than global contrast.")
