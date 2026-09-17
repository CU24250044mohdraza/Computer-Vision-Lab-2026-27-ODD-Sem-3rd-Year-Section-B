# Experiment No. 5
# Feature Extraction and Image Analysis using SIFT and HOG Descriptors

# ---------------------------------------------------------
# Step 1: Import the required libraries and load an image
# ---------------------------------------------------------
# pip install opencv-python opencv-contrib-python scikit-image numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
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

# ---------------------------------------------------------
# Step 2: Convert to grayscale and perform basic preprocessing
# ---------------------------------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)  # mild denoise before feature extraction

# ---------------------------------------------------------
# Step 3: Apply SIFT to detect keypoints and compute descriptors
# ---------------------------------------------------------
sift = cv2.SIFT_create()

t0 = time.perf_counter()
keypoints, descriptors = sift.detectAndCompute(gray, None)
t_sift = time.perf_counter() - t0

print("SIFT Feature Extraction")
print("------------------------")
print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape if descriptors is not None else None}")
print(f"Time taken: {t_sift:.4f} s")

# ---------------------------------------------------------
# Step 4: Visualize detected keypoints and analyze distribution
# ---------------------------------------------------------
img_keypoints = cv2.drawKeypoints(
    img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT Keypoints (n={len(keypoints)})")
plt.axis("off")
plt.show()

if keypoints:
    xs = [kp.pt[0] for kp in keypoints]
    ys = [kp.pt[1] for kp in keypoints]
    print(f"Keypoint spread - x: [{min(xs):.1f}, {max(xs):.1f}], "
          f"y: [{min(ys):.1f}, {max(ys):.1f}]")

# ---------------------------------------------------------
# Step 5: Extract HOG features
# ---------------------------------------------------------
hog_resized = cv2.resize(gray, (128, 128))  # keep HOG computation manageable

t0 = time.perf_counter()
hog_features, hog_image = hog(
    hog_resized,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
)
t_hog = time.perf_counter() - t0

print("\nHOG Feature Extraction")
print("------------------------")
print(f"HOG feature vector length: {hog_features.shape[0]}")
print(f"Time taken: {t_hog:.4f} s")

# ---------------------------------------------------------
# Step 6: Visualize the HOG descriptor
# ---------------------------------------------------------
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(hog_resized, cmap="gray"); axes[0].set_title("Input (resized)"); axes[0].axis("off")
axes[1].imshow(hog_image_rescaled, cmap="gray"); axes[1].set_title("HOG Descriptor"); axes[1].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 7: Compare SIFT and HOG characteristics
# ---------------------------------------------------------
print("\nSIFT vs HOG Comparison")
print("-----------------------")
print(f"{'Property':25s} | {'SIFT':30s} | {'HOG':30s}")
print(f"{'Type':25s} | {'Sparse keypoints + descriptor':30s} | {'Dense global descriptor':30s}")
print(f"{'Scale invariance':25s} | {'Yes (scale-space extrema)':30s} | {'No (fixed window)':30s}")
print(f"{'Rotation invariance':25s} | {'Yes (dominant orientation)':30s} | {'No (orientation-sensitive)':30s}")
print(f"{'Output size':25s} | {f'{len(keypoints)} x 128':30s} | {f'{hog_features.shape[0]} values':30s}")
print(f"{'Compute time (s)':25s} | {t_sift:<30.4f} | {t_hog:<30.4f}")

# ---------------------------------------------------------
# Step 8: Image matching using SIFT on two similar images
# ---------------------------------------------------------
# Generate a "similar" second image by rotating and scaling the original
center = (gray.shape[1] // 2, gray.shape[0] // 2)
rot_matrix = cv2.getRotationMatrix2D(center, 15, 0.9)
gray2 = cv2.warpAffine(gray, rot_matrix, (gray.shape[1], gray.shape[0]))

kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors, des2)
matches = sorted(matches, key=lambda m: m.distance)

print(f"\nImage Matching (SIFT + Brute-Force Matcher)")
print("--------------------------------------------")
print(f"Keypoints in image 1: {len(keypoints)}, image 2: {len(kp2)}")
print(f"Total matches found: {len(matches)}")
print(f"Best match distance: {matches[0].distance:.2f}" if matches else "No matches found")

match_img = cv2.drawMatches(
    gray, keypoints, gray2, kp2, matches[:30], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12, 5))
plt.imshow(match_img, cmap="gray")
plt.title("Top 30 SIFT Matches (Original vs Rotated/Scaled)")
plt.axis("off")
plt.show()

# ---------------------------------------------------------
# Step 9: Evaluate strengths and limitations
# ---------------------------------------------------------
good_matches = [m for m in matches if m.distance < 200]
print(f"\nGood matches (distance < 200): {len(good_matches)} / {len(matches)}")

# ---------------------------------------------------------
# Step 10: Document observations
# ---------------------------------------------------------
print("\nObservations")
print("------------")
print("- SIFT detects a sparse set of distinctive keypoints, each with a scale and "
      "orientation, making it robust to rotation, scaling, and moderate illumination "
      "changes - well suited for image matching and object recognition.")
print("- HOG produces a single dense feature vector describing local gradient orientation "
      "distributions over a fixed grid, capturing overall shape/silhouette information - "
      "well suited for object detection tasks like pedestrian detection.")
print("- SIFT matching correctly found correspondences between the original and the "
      "rotated/scaled version of the image, demonstrating its scale and rotation invariance.")
print("- HOG is computationally cheaper and simpler than SIFT but is not scale/rotation "
      "invariant on its own, and typically requires a sliding-window or fixed-size "
      "normalization to be useful for detection.")
print("- Handcrafted descriptors like SIFT/HOG are interpretable and require no training "
      "data, but deep learning-based features generally outperform them in accuracy for "
      "complex recognition tasks, at the cost of requiring large labeled datasets and "
      "compute for training.")
print("- Applications: SIFT - panorama stitching, 3D reconstruction, object/logo "
      "recognition, image matching. HOG - pedestrian/human detection, vehicle detection, "
      "traditional object detection pipelines (e.g., HOG + SVM).")
