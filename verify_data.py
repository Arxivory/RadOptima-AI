import matplotlib.pyplot as plt
from data.dicom_loader import load_dicom_volume

DATA_PATH = "data/samples/test_scan"

try:
	volume = load_dicom_volume(DATA_PATH)

	mid_slice_idx = volume.shape[0] //2
	mid_slice = volume[mid_slice_idx, :, :]

	plt.figure(figsize=(10, 10))

	plt.imshow(mid_slice, cmap='bone', vmin=-200, vmax=500)
	plt.title(f"Chest Slice {mid_slice_idx} - HU Normalized")
	plt.colorbar(label="Housefield Units (HU)")
	plt.show()
	
except Exception as e:
	print(f"Error {e}")