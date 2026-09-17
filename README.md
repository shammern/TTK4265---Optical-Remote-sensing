# TTK4265 - Optical Remote Sensing

## Scripts

- `image_merger.py` - colorizes and blends the argon (red) and mercury (green) spectra into one overlap image.
- `spectogram_calibration.py` - fits pixel-to-wavelength calibration models and compares them with leave-k-out RMSE.
- `overlay_calibration_wavelengths.py` - overlays the calibration wavelengths on the merged spectral image.
- `repeated_spectra_uncertainty.py` - computes pixel and spectrum repeatability uncertainty from the 10 repeated frames per lamp.
- `warmup_investigation.py` - checks whether lamp intensity drifts over the acquisition, i.e. whether warm-up affects the readings.
