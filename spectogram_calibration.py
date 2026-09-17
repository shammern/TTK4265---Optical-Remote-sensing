from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

wavelength = np.array([435.83, 546.07, (576.96+579.06)/2, 696.54, 706.72, 763.51]) # First 3 values are mercury lines(the third is a dublet), last 3 are argon lines.
pixels = np.array([347, 647, 735, 1064, 1092, 1255]) # Pixel values found by manual inspection

image_pixels = np.linspace(pixels.min(), pixels.max(), 1000)

leave_out = 0 # Number of points to leave out when interpolating. 
              # Cubic spline will always fit all points so it will always give RMSE = 0 without leave out


def polynomial_model(train_pixels: np.ndarray, train_wavelength: np.ndarray, degree: int):
	coefficients = np.polyfit(train_pixels, train_wavelength, degree)
	return lambda values: np.polyval(coefficients, values)


def build_models(train_pixels: np.ndarray, train_wavelength: np.ndarray) -> list[tuple[str, object]]:
	return [
		(
			"Linear",
			polynomial_model(train_pixels, train_wavelength, degree=1),
		),
		(
			"Quadratic",
			polynomial_model(train_pixels, train_wavelength, degree=2),
		),
		(
			"Cubic",
			polynomial_model(train_pixels, train_wavelength, degree=3),
		),
		(
			"Cubic spline",
			CubicSpline(train_pixels, train_wavelength),
		),
	]


def print_model_formulas(
	train_pixels: np.ndarray, train_wavelength: np.ndarray
) -> None:
	print("\nCalibration formulas")
	for name, degree in (("Linear", 1), ("Quadratic", 2), ("Cubic", 3)):
		coefficients = np.polyfit(train_pixels, train_wavelength, degree)
		print(f"{name} coefficients (highest power first):")
		print(", ".join(f"{coefficient:.12g}" for coefficient in coefficients))
		print(
			f"  lambda(p) = "
			+ " + ".join(
				f"({coefficient:.12g})*p^{degree - index}"
				for index, coefficient in enumerate(coefficients)
			)
		)

	spline = CubicSpline(train_pixels, train_wavelength)
	print("Cubic spline coefficients:")
	print("Each segment uses d = p - left_pixel and c[0:4, i].")
	for index, left_pixel in enumerate(train_pixels[:-1]):
		cubic, quadratic, linear, constant = spline.c[:, index]
		print(
			f"  {left_pixel:g} <= p <= {train_pixels[index + 1]:g}: "
			f"lambda(p) = ({cubic:.12g})*d^3 + "
			f"({quadratic:.12g})*d^2 + "
			f"({linear:.12g})*d + "
			f"({constant:.12g})"
		)


def leave_out_predictions(leave_out: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
	model_names = [name for name, _ in build_models(pixels, wavelength)]

	if leave_out == 0:
		return {
			name: (
				wavelength.copy(),
				model(pixels),
				model(pixels),
			)
			for name, model in build_models(pixels, wavelength)
		}

	actual_values = {name: [] for name in model_names}
	predicted_values = {name: [] for name in model_names}
	point_predictions = {name: [[] for _ in pixels] for name in model_names}

	for omitted_indices in combinations(range(len(pixels)), leave_out):
		training_mask = np.ones(len(pixels), dtype=bool)
		training_mask[list(omitted_indices)] = False
		models_without_point = build_models(
			pixels[training_mask], wavelength[training_mask]
		)
		for name, model in models_without_point:
			predictions = np.asarray(model(pixels[list(omitted_indices)]))
			actual_values[name].extend(wavelength[list(omitted_indices)])
			predicted_values[name].extend(predictions)
			for index, prediction in zip(omitted_indices, predictions):
				point_predictions[name][index].append(prediction)

	return {
		name: (
			np.asarray(actual_values[name]),
			np.asarray(predicted_values[name]),
			np.array(
				[
					np.mean(predictions) if predictions else np.nan
					for predictions in point_predictions[name]
				]
			),
		)
		for name in model_names
	}



def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
	return float(np.sqrt(np.mean((actual - predicted) ** 2)))


full_data_models = build_models(pixels, wavelength)
validation_results = leave_out_predictions(leave_out)
print_model_formulas(pixels, wavelength)

print(f"Leave-{leave_out}-out model comparison")

figure, (calibration_axis, residual_axis) = plt.subplots(
	2, 1, figsize=(11, 9), sharex=True
)

for number, (name, full_data_model) in enumerate(full_data_models, start=1):
	model_curve = full_data_model(image_pixels)
	actual_values, predicted_values, mean_predictions = validation_results[name]
	residuals = wavelength - mean_predictions
	rmse = calculate_rmse(actual_values, predicted_values)

	print(f"{number}. {name}: RMSE = {rmse:.6f} nm")
	calibration_axis.plot(image_pixels, model_curve, label=name)
	residual_axis.plot(pixels, residuals, "o-", label=name)

calibration_axis.scatter(
	pixels,
	wavelength,
	color="black",
	s=45,
	label="Measured calibration lines",
	zorder=3,
)
calibration_axis.set_ylabel("Wavelength [nm]")
calibration_axis.set_title(
	f"Leave-{leave_out}-out pixel-to-wavelength comparison"
)
calibration_axis.legend(fontsize="small", ncol=2)
calibration_axis.grid(True, alpha=0.3)

residual_axis.axhline(0, color="black", linewidth=0.8)
residual_axis.set_xlabel("Pixel position")
residual_axis.set_ylabel("Held-out residual [nm]")
residual_axis.set_title(
	"Measured wavelength minus mean held-out prediction"
)
residual_axis.legend(fontsize="small")
residual_axis.grid(True, alpha=0.3)

figure.tight_layout()
plt.show()








