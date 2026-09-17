# Experiment No. 4
# Frequency Domain Image Filtering using Fourier Transform

# ---------------------------------------------------------
# Step 1: Import the required libraries and load a grayscale image
# ---------------------------------------------------------
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
rows, cols = gray.shape

# ---------------------------------------------------------
# Step 2: Compute the Discrete Fourier Transform (DFT)
# ---------------------------------------------------------
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)

# ---------------------------------------------------------
# Step 3: Shift the zero-frequency component to the center
# ---------------------------------------------------------
dft_shift = np.fft.fftshift(dft)

# ---------------------------------------------------------
# Step 4: Display and analyze the magnitude spectrum
# ---------------------------------------------------------
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Original Grayscale Image"); axes[0].axis("off")
axes[1].imshow(magnitude_spectrum, cmap="gray"); axes[1].set_title("Magnitude Spectrum"); axes[1].axis("off")
plt.tight_layout()
plt.show()

print("Frequency Spectrum Stats")
print("-------------------------")
print(f"Magnitude spectrum range: [{magnitude_spectrum.min():.2f}, {magnitude_spectrum.max():.2f}]")

# ---------------------------------------------------------
# Step 5: Design and apply a Low-Pass Frequency Filter
# ---------------------------------------------------------
def circular_mask(shape, radius, low_pass=True):
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - center_r) ** 2 + (x - center_c) ** 2)
    mask = (dist <= radius).astype(np.float32)
    if not low_pass:
        mask = 1 - mask
    return np.stack([mask, mask], axis=-1)

RADIUS = 30

lp_mask = circular_mask((rows, cols), RADIUS, low_pass=True)
lp_dft_shift = dft_shift * lp_mask

# ---------------------------------------------------------
# Step 6: Design and apply a High-Pass Frequency Filter
# ---------------------------------------------------------
hp_mask = circular_mask((rows, cols), RADIUS, low_pass=False)
hp_dft_shift = dft_shift * hp_mask

# ---------------------------------------------------------
# Step 7: Perform the Inverse Fourier Transform (IDFT)
# ---------------------------------------------------------
def reconstruct(shifted_dft):
    unshifted = np.fft.ifftshift(shifted_dft)
    reconstructed = cv2.idft(unshifted)
    reconstructed = cv2.magnitude(reconstructed[:, :, 0], reconstructed[:, :, 1])
    return cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

low_pass_result = reconstruct(lp_dft_shift)
high_pass_result = reconstruct(hp_dft_shift)

# ---------------------------------------------------------
# Step 8: Compare original, spectrum, low-pass, and high-pass results
# ---------------------------------------------------------
lp_spectrum = 20 * np.log(cv2.magnitude(lp_dft_shift[:, :, 0], lp_dft_shift[:, :, 1]) + 1)
hp_spectrum = 20 * np.log(cv2.magnitude(hp_dft_shift[:, :, 0], hp_dft_shift[:, :, 1]) + 1)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes[0, 0].imshow(gray, cmap="gray"); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
axes[0, 1].imshow(magnitude_spectrum, cmap="gray"); axes[0, 1].set_title("Full Spectrum"); axes[0, 1].axis("off")
axes[0, 2].imshow(lp_spectrum, cmap="gray"); axes[0, 2].set_title("Low-Pass Spectrum"); axes[0, 2].axis("off")
axes[0, 3].imshow(hp_spectrum, cmap="gray"); axes[0, 3].set_title("High-Pass Spectrum"); axes[0, 3].axis("off")
axes[1, 0].imshow(gray, cmap="gray"); axes[1, 0].set_title("Original"); axes[1, 0].axis("off")
axes[1, 1].axis("off")
axes[1, 2].imshow(low_pass_result, cmap="gray"); axes[1, 2].set_title("Low-Pass Filtered"); axes[1, 2].axis("off")
axes[1, 3].imshow(high_pass_result, cmap="gray"); axes[1, 3].set_title("High-Pass Filtered"); axes[1, 3].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 9: Analyze the impact on smoothness, edges, and noise
# ---------------------------------------------------------
print("\nQuantitative Comparison")
print("-----------------------")
print(f"{'Image':20s} | {'Mean':8s} | {'Std':8s}")
print(f"{'Original':20s} | {gray.mean():8.2f} | {gray.std():8.2f}")
print(f"{'Low-Pass Filtered':20s} | {low_pass_result.mean():8.2f} | {low_pass_result.std():8.2f}")
print(f"{'High-Pass Filtered':20s} | {high_pass_result.mean():8.2f} | {high_pass_result.std():8.2f}")
print(f"\nLow-pass cutoff radius: {RADIUS} px (retains low frequencies -> smoothing)")
print(f"High-pass cutoff radius: {RADIUS} px (removes low frequencies -> edge/detail emphasis)")

# ---------------------------------------------------------
# Step 10: Document observations
# ---------------------------------------------------------
print("\nObservations")
print("------------")
print("- The magnitude spectrum shows most energy concentrated near the center (low "
      "frequencies), corresponding to smooth intensity regions of the image.")
print("- Shifting the zero-frequency component to the center makes the spectrum symmetric "
      "and easier to interpret and filter with radially symmetric masks.")
print("- The Low-Pass Filter retains only low frequencies near the center, producing a "
      "smoothed/blurred reconstruction similar to spatial Gaussian blurring, reducing noise "
      "and fine detail.")
print("- The High-Pass Filter removes low frequencies and keeps only high-frequency "
      "components, producing an edge/detail-emphasized reconstruction while losing overall "
      "brightness and smooth regions.")
print("- Frequency domain filtering separates smoothing and edge enhancement cleanly via "
      "an explicit cutoff radius, which can be more intuitive to design than spatial kernels, "
      "though it requires forward/inverse transforms and is often costlier for small images.")
print("- Applications: medical image enhancement, satellite image analysis, image "
      "restoration, and biometric (fingerprint/iris) systems.")
