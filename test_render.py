import glfw
import numpy as np
from data.dicom_loader import load_dicom_volume
import radoptima_core

def main():
	if not glfw.init():
		return

	window = glfw.create_window(512, 512, "RadOptima Engine Test", None, None)
	if not window:
		glfw.terminate()
		return

	glfw.make_context_current(window)

	engine = radoptima_core.RadEngine()
	engine.init_opengl()

	data_path = "data/samples/test_scan"
	volume = load_dicom_volume(data_path)

	engine.upload_volume(volume)

	print("Pipeline Test Complete. The volume is not living in VRAM.")

	glfw.terminate()

if __name__ == "__main__":
	main()