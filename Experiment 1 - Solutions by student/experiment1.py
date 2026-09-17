# Experiment No. 1
# Implementation of Fundamental Image Processing Operations using Python and OpenCV

# ---------------------------------------------------------
# Step 1: Install and import the required libraries
# ---------------------------------------------------------
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_PATH = "input.jpeg"  # replace with your image path

# ---------------------------------------------------------
# Step 2: Load a color image and display it using OpenCV and Matplotlib
# ---------------------------------------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    # Fallback: generate a sample color image if no file is found
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (180, 150), (0, 0, 255), -1)      # red
    cv2.rectangle(img, (200, 50), (350, 200), (0, 255, 0), -1)     # green
    cv2.circle(img, (200, 250), 60, (255, 0, 0), -1)               # blue
    cv2.imwrite(IMAGE_PATH, img)
    print(f"No image found at '{IMAGE_PATH}'. Generated a sample image instead.")

# Display with OpenCV
cv2.imshow("Original Image (OpenCV)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display with Matplotlib (convert BGR -> RGB first)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 4))
plt.imshow(img_rgb)
plt.title("Original Image (Matplotlib)")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 3: Examine image properties
# ---------------------------------------------------------
height, width, channels = img.shape
print("Image Properties")
print("-----------------")
print(f"Dimensions (H x W): {height} x {width}")
print(f"Number of channels: {channels}")
print(f"Data type: {img.dtype}")
print(f"Total pixels: {img.size}")

# ---------------------------------------------------------
# Step 4: Save the image in different formats and compare
# ---------------------------------------------------------
cv2.imwrite("output.jpg", img)
cv2.imwrite("output.png", img)

jpg_size = os.path.getsize("output.jpg")
png_size = os.path.getsize("output.png")
print(f"\nJPEG file size: {jpg_size} bytes")
print(f"PNG file size: {png_size} bytes")

# ---------------------------------------------------------
# Step 5: Convert to Grayscale, HSV, and LAB color spaces
# ---------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(gray, cmap="gray"); axes[1].set_title("Grayscale"); axes[1].axis("off")
axes[2].imshow(hsv); axes[2].set_title("HSV"); axes[2].axis("off")
axes[3].imshow(lab); axes[3].set_title("LAB"); axes[3].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 6: Geometric transformations - resize, rotation, flipping
# ---------------------------------------------------------
resized = cv2.resize(img, (width // 2, height // 2))

center = (width // 2, height // 2)
rot_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, rot_matrix, (width, height))

flipped_h = cv2.flip(img, 1)  # horizontal
flipped_v = cv2.flip(img, 0)  # vertical

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)); axes[0].set_title("Resized"); axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)); axes[1].set_title("Rotated 45°"); axes[1].axis("off")
axes[2].imshow(cv2.cvtColor(flipped_h, cv2.COLOR_BGR2RGB)); axes[2].set_title("Flipped Horizontal"); axes[2].axis("off")
axes[3].imshow(cv2.cvtColor(flipped_v, cv2.COLOR_BGR2RGB)); axes[3].set_title("Flipped Vertical"); axes[3].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 7: Generate the complement (negative) of the image
# ---------------------------------------------------------
negative = 255 - img

plt.figure(figsize=(5, 4))
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative (Complement) Image")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 8: Crop a Region of Interest (ROI) and analyze it
# ---------------------------------------------------------
x1, y1, x2, y2 = width // 4, height // 4, 3 * width // 4, 3 * height // 4
roi = img[y1:y2, x1:x2]

print("\nROI Properties")
print("--------------")
print(f"ROI shape: {roi.shape}")
print(f"ROI mean pixel value (per channel, BGR): {roi.mean(axis=(0, 1))}")

plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("Cropped ROI")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 9: Display original and processed images together for comparison
# ---------------------------------------------------------
titles = ["Original", "Grayscale", "HSV", "LAB", "Resized", "Rotated",
          "Flipped H", "Flipped V", "Negative", "ROI"]
images = [img_rgb, gray, hsv, lab,
          cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(flipped_h, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(flipped_v, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(negative, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, title, image in zip(axes.ravel(), titles, images):
    cmap = "gray" if image.ndim == 2 else None
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 10: Document observations
# ---------------------------------------------------------
print("\nObservations")
print("------------")
print("- Grayscale conversion reduces the image to a single intensity channel, "
      "removing color information but preserving structure.")
print("- HSV separates color (hue) from intensity, useful for color-based segmentation.")
print("- LAB separates lightness from color, useful for perceptually uniform processing.")
print("- Resizing/rotation/flipping alter geometry without necessarily losing information.")
print("- The negative image highlights details in dark regions by inverting intensities.")
print("- Cropping an ROI isolates a region for focused analysis, reducing computation "
      "for subsequent tasks.")
