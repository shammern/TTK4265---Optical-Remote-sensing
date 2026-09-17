from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

elements = ["AR", "HG"]
folder = Path("Data") / "Spectral"
row_band = slice(580, 620)  # Narrow band around the calibration row (y=600) to limit smile-induced blurring.

figure, axes = plt.subplots(2, 2, figsize=(12, 8))

for row, element in enumerate(elements):
    files = sorted(folder.glob(f"SpecCal_{element}_800exp_*.png"))

    images = []
    for file in files:
        with Image.open(file) as image:
            images.append(np.asarray(image.convert("L"), dtype=float))

    images = np.stack(images)
    std_image = np.std(images, axis=0, ddof=1)

    # Average over a narrow row band to produce one spectrum for each repeated frame.
    spectra = images[:, row_band, :].mean(axis=1)
    std_spectrum = spectra.std(axis=0, ddof=1)
    pixels = np.arange(images.shape[2])

    print(f"{element} images: {len(files)}")
    print(f"{element} image size: {images.shape[2]} x {images.shape[1]} pixels")
    print(f"{element} mean pixel uncertainty: {std_image.mean():.4f} counts")
    print(f"{element} maximum pixel uncertainty: {std_image.max():.4f} counts")
    print(f"{element} mean spectrum uncertainty: {std_spectrum.mean():.4f} counts")

    axes[row, 0].plot(pixels, std_spectrum, color="tab:red")
    axes[row, 0].set_xlabel("Pixel position")
    axes[row, 0].set_ylabel("Standard deviation [counts]")
    axes[row, 0].set_title(f"{element}: spectral repeatability uncertainty")
    axes[row, 0].grid(alpha=0.3)

    axes[row, 1].imshow(std_image, cmap="magma", aspect="auto")
    axes[row, 1].set_xlabel("Spectral pixel")
    axes[row, 1].set_ylabel("Spatial pixel")
    axes[row, 1].set_title(f"{element}: difference between frames")

figure.tight_layout()
plt.show()
