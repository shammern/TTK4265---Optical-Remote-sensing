from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Same calibration points as spectogram_calibration.py: first 3 are mercury
# lines (the third is the 576.96/579.06 nm doublet), last 3 are argon lines.
wavelength = np.array([435.83, 546.07, (576.96 + 579.06) / 2, 696.54, 706.72, 763.51])
pixels = np.array([347, 647, 735, 1064, 1092, 1255])

image_path = Path("Results") / "spectral_blended.png"
with Image.open(image_path) as image:
    blended = np.asarray(image)

# Quadratic pixel-to-wavelength model, used only to label the top axis;
# the calibration lines themselves are drawn at their measured pixel values.
coefficients = np.polyfit(pixels, wavelength, 2)
pixel_to_wavelength = lambda p: np.polyval(coefficients, p)

figure, image_axis = plt.subplots(figsize=(14, 6))
image_axis.imshow(blended, aspect="auto")
image_axis.set_xlabel("Pixel position")
image_axis.set_ylabel("Spatial pixel")
image_axis.set_title("Merged spectral image with calibration wavelengths")

label_transform = image_axis.get_xaxis_transform()  # x in data coords, y in axes fraction
for pixel, wl in zip(pixels, wavelength):
    image_axis.axvline(pixel, color="white", linestyle="--", linewidth=1, alpha=0.8)
    image_axis.text(
        pixel,
        -0.03,
        f"{wl:.2f} nm",
        transform=label_transform,
        color="black",
        rotation=90,
        ha="right",
        va="top",
        fontsize=8,
        clip_on=False,
    )

wavelength_axis = image_axis.secondary_xaxis(
    "top", functions=(pixel_to_wavelength, lambda wl: np.interp(wl, pixel_to_wavelength(np.arange(blended.shape[1])), np.arange(blended.shape[1])))
)
wavelength_axis.set_xlabel("Wavelength [nm] (quadratic calibration)")

figure.subplots_adjust(bottom=0.28, top=0.85)
plt.show()
