import pydicom
import numpy as np
import os

def load_dicom_volume(directory):
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.DCM')]
    
    if not file_list:
        raise ValueError(f"No DICOM files found in {directory}")

    slices = [pydicom.dcmread(f) for f in file_list]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    pixel_spacing = slices[0].PixelSpacing

    z_spacing = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])

    total_z = z_spacing * len(slices)
    total_y = slices[0].Rows * pixel_spacing[0]
    total_x = slices[0].Columns * pixel_spacing[1]

    max_dim = max(total_x, total_y, total_z)
    scale_vector = (total_x / max_dim, total_y / max_dim, total_z / max_dim)

    volume = np.stack([s.pixel_array for s in slices])

    for i, s in enumerate(slices):
        slope = float(s.RescaleSlope) if 'RescaleSlope' in s else 1.0
        intercept = float(s.RescaleIntercept) if 'RescaleIntercept' in s else -1024.0
        volume[i] = (volume[i].astype(np.float32) * slope + intercept)

    print(f"Loaded Volume Shape: {volume.shape} (Z, Y, X)")
    return volume.astype(np.int16), scale_vector