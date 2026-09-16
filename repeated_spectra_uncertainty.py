from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

element = "HG" # AR/HG
folder = Path("Data") / "Spectral"
files = sorted(folder.glob(f"SpecCal_{element}_800exp_*.png"))

images = []
for file in files:
    with Image.open(file) as image:
        images.append(np.asarray(image.convert("L"), dtype=float))

images = np.stack(images)
mean_image = np.mean(images, axis=0)
std_image = np.std(images, axis=0, ddof=1)

# Average over rows to produce one spectrum for each repeated frame.
spectra = images.mean(axis=1)
mean_spectrum = spectra.mean(axis=0)
std_spectrum = spectra.std(axis=0, ddof=1)
pixels = np.arange(images.shape[2])

print(f"Images: {len(files)}")
print(f"Image size: {images.shape[2]} x {images.shape[1]} pixels")
print(f"Mean pixel uncertainty: {std_image.mean():.4f} counts")
print(f"Maximum pixel uncertainty: {std_image.max():.4f} counts")
print(f"Mean spectrum uncertainty: {std_spectrum.mean():.4f} counts")

figure, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(pixels, mean_spectrum, color="black", label="Mean spectrum")
axes[0, 0].fill_between(
    pixels,
    mean_spectrum - std_spectrum,
    mean_spectrum + std_spectrum,
    color="tab:blue",
    alpha=0.3,
    label="±1 standard deviation",
)
axes[0, 0].set_ylabel("Intensity [counts]")
axes[0, 0].set_title(f"{element}: mean spectrum")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(pixels, std_spectrum, color="tab:red")
axes[0, 1].set_xlabel("Pixel position")
axes[0, 1].set_ylabel("Standard deviation [counts]")
axes[0, 1].set_title("Spectral repeatability uncertainty")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].imshow(mean_image, cmap="gray", aspect="auto")
axes[1, 0].set_xlabel("Spectral pixel")
axes[1, 0].set_ylabel("Spatial pixel")
axes[1, 0].set_title("Mean image: overlap of all frames")

axes[1, 1].imshow(std_image, cmap="magma", aspect="auto")
axes[1, 1].set_xlabel("Spectral pixel")
axes[1, 1].set_ylabel("Spatial pixel")
axes[1, 1].set_title("Difference between frames")

figure.tight_layout()
plt.show()
