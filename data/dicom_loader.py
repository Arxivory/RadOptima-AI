import pydicom
import numpy as np
import os

def load_dicom_volume(directory):
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    
    if not file_list:
        raise ValueError(f"No DICOM files found in {directory}")

    slices = [pydicom.dcmread(f) for f in file_list]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices])

    for i, s in enumerate(slices):
        slope = float(s.RescaleSlope) if 'RescaleSlope' in s else 1.0
        intercept = float(s.RescaleIntercept) if 'RescaleIntercept' in s else -1024.0
        volume[i] = (volume[i].astype(np.float32) * slope + intercept)

    print(f"Loaded Volume Shape: {volume.shape} (Z, Y, X)")
    return volume.astype(np.int16)