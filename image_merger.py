from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path
import matplotlib.pyplot as plt

argon_path = Path("Data") / "Spectral" / "SpecCal_AR_800exp_1.png"
mercury_path = Path("Data") / "Spectral" / "SpecCal_HG_800exp_1.png"

argon_image_raw = Image.open(argon_path)
mercury_image_raw = Image.open(mercury_path)

argon_grayscale = argon_image_raw.convert("L")
mercury_grayscale = mercury_image_raw.convert("L")

argon_color = ImageOps.colorize(argon_grayscale, black="black", white="red")
mercury_color = ImageOps.colorize(mercury_grayscale, black="black", white="green")

mercury_color.save(Path("Data") / "Spectral" / "argon_color.png")


argon_color.show()
mercury_color.show()

mix = Image.blend(argon_color, mercury_color, 0.5)

brightness = ImageEnhance.Brightness(mix)
mix_brighter = brightness.enhance(2)

mix_brighter.show()

plt.imshow(mercury_image_raw)
plt.show()