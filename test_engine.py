import sys
import os
import numpy as np

build_path = os.path.join(os.getcwd(), "out", "build", "x64-Debug", "cpp_core")

if os.path.exists(build_path):
    sys.path.append(build_path)
    print(f"Added build path: {build_path}")
else:
    print(f"Error: Could not find build folder at {build_path}")
    print("Make sure you have clicked 'Build All' in Visual Studio.")

from data.dicom_loader import load_dicom_volume

try:
    import radoptima_core

    
    print(f"I am loading the engine from: {radoptima_core.__file__}")

    data_path = "data/samples/test_scan"
    volume = load_dicom_volume(data_path)

    engine = radoptima_core.RadEngine()

    engine.upload_volume(volume)

    print("Handshake successful! C++ has access to the DICOM pixels.")
    
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")