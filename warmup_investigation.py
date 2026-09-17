import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

elements = ["AR", "HG"]
folder = Path("Data") / "Spectral"


def frame_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.png$", path.name)
    return int(match.group(1))


figure, axes = plt.subplots(2, 2, figsize=(12, 8))

for row, element in enumerate(elements):
    files = sorted(folder.glob(f"SpecCal_{element}_800exp_*.png"), key=frame_index)

    # Real acquisition time from file timestamps, relative to the first frame.
    timestamps = np.array([file.stat().st_mtime for file in files])
    elapsed = timestamps - timestamps[0]

    spectra = []
    for file in files:
        with Image.open(file) as image:
            data = np.asarray(image.convert("L"), dtype=float)
        spectra.append(data.mean(axis=0))
    spectra = np.stack(spectra)

    total_intensity = spectra.sum(axis=1)
    peak_intensity = spectra.max(axis=1)

    # Linear trend of total intensity vs elapsed time: a warm-up effect shows
    # up as a significant monotonic slope, not just random scatter.
    slope, intercept = np.polyfit(elapsed, total_intensity, 1)
    fit = slope * elapsed + intercept
    residuals = total_intensity - fit
    r2 = 1 - np.sum(residuals**2) / np.sum((total_intensity - total_intensity.mean()) ** 2)
    correlation = np.corrcoef(elapsed, total_intensity)[0, 1]
    percent_change = 100 * (total_intensity[-1] - total_intensity[0]) / total_intensity[0]

    print(f"{element}:")
    print(f"  Total acquisition time: {elapsed[-1]:.1f} s over {len(files)} frames")
    print(f"  Total intensity change first->last frame: {percent_change:+.2f} %")
    print(f"  Linear trend slope: {slope:+.3f} counts/s (R^2 = {r2:.3f}, r = {correlation:+.3f})")
    print(f"  Peak line intensity change first->last frame: "
          f"{100 * (peak_intensity[-1] - peak_intensity[0]) / peak_intensity[0]:+.2f} %")
    print()

    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    pixels = np.arange(spectra.shape[1])
    for spectrum, color in zip(spectra, colors):
        axes[row, 0].plot(pixels, spectrum, color=color, alpha=0.7)
    axes[row, 0].set_xlabel("Pixel position")
    axes[row, 0].set_ylabel("Intensity [counts]")
    axes[row, 0].set_title(f"{element}: spectra over time (dark→light = early→late)")
    axes[row, 0].grid(alpha=0.3)

    axes[row, 1].scatter(elapsed, total_intensity, color="tab:blue", label="Total intensity")
    axes[row, 1].plot(elapsed, fit, color="tab:red", linestyle="--",
                       label=f"Linear fit (r={correlation:+.2f})")
    axes[row, 1].set_xlabel("Elapsed time [s]")
    axes[row, 1].set_ylabel("Total spectrum intensity [counts]")
    axes[row, 1].set_title(f"{element}: intensity drift over acquisition")
    axes[row, 1].legend()
    axes[row, 1].grid(alpha=0.3)

figure.tight_layout()
plt.show()
