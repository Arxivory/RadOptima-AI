import sys
import os
import glfw
import numpy as np
import ctypes
from ctypes import wintypes
from data.dicom_loader import load_dicom_volume
from scipy.ndimage import median_filter
import shutil
from ldctbench.hub import Methods
from ldctbench.hub.utils import denoise_dicom
import torch
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
from pathlib import Path

build_path = os.path.join(os.getcwd(), "out", "build", "x64-Debug", "cpp_core")

if os.path.exists(build_path):
    sys.path.append(build_path)
    print(f"Added build path: {build_path}")
else:
    print(f"Error: Could not find build folder at {build_path}")
    print("Make sure you have clicked 'Build All' in Visual Studio.")


import radoptima_core

last_x, last_y = 256, 256
first_mouse = True

def convert_to_explicit_vr(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Converting DICOMs to Explicit VR for AI compatibility...")
    for f in os.listdir(input_dir):
        if f.endswith('.DCM') or f.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(input_dir, f))
            
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.is_implicit_VR = False
            ds.is_little_endian = True
            
            ds.save_as(os.path.join(output_dir, f))

def get_ai_enhanced_volume(original_dicom_path):
    temp_ai_dir = "./data/temp_ai_out"
    if os.path.exists(temp_ai_dir):
        shutil.rmtree(temp_ai_dir) 
    os.makedirs(temp_ai_dir)

    print("AI is reconstructing the volume via RED-CNN...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    denoise_dicom(
        dicom_path=original_dicom_path,
        savedir=temp_ai_dir,
        method=Methods.REDCNN,
        device=device
    )

def has_ai_output(ai_data_path):
	if Path(ai_data_path).exists() and any(f.endswith('.DCM') or f.endswith('.dcm') for f in os.listdir(ai_data_path)):
		return True

	return False

def main():
	global first_mouse, last_x, last_y

	if not glfw.init():
		return

	glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
	glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
	glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

	title = "RadOptima Engine Test"
	window = glfw.create_window(1280, 720, title, None, None)
	if not window:
		glfw.terminate()
		return

	glfw.make_context_current(window)

	engine = radoptima_core.RadEngine()

	try:
		engine.init_opengl()
		print("GLAD Initialized successfully!")
	except Exception as e:
		print(f"C++ Error: {e}")
		return

	hwnd = ctypes.windll.user32.FindWindowW(None, title)
	if not hwnd:
		print("Error: Could not find the window handle via Win32 API.")
		return
		
	engine.init_imgui(hwnd)

	engine.setup_cube()

	data_path = "data/samples/chest"
	explicit_vr_path = "data/explicit_vr"
	ai_data_path = "data/temp_ai_out"

	volume, volume_scale = load_dicom_volume(data_path)

	if not has_ai_output(ai_data_path):
		print("Converting...")
		convert_to_explicit_vr(data_path, explicit_vr_path)

		print("No AI output found. Running AI enhancement...")
		get_ai_enhanced_volume(explicit_vr_path)

	print("AI output found. Loading AI-enhanced volume...")

	ai_volume, ai_volume_scale = load_dicom_volume(ai_data_path)

	engine.upload_volume(volume)
	engine.upload_ai_volume(ai_volume)

	tf_data = np.zeros((256, 4), dtype=np.float32)
	for i in range(256):
		intensity = i / 255.0
		tf_data[i] = [intensity, intensity * 0.5, 0.2, intensity] 
		
	engine.update_transfer_function(tf_data)

	with open("shaders/raymarch.vert", "r") as f: v_src = f.read()
	with open("shaders/raymarch.frag", "r") as f: f_src = f.read()
	engine.compile_shader(v_src, f_src)

	engine.set_window_level(400, 40)

	current_slice = volume.shape[0] // 2
	depth = volume.shape[0]

	while not glfw.window_should_close(window):
		glfw.poll_events()
		width, height = glfw.get_framebuffer_size(window)
		
		if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
			current_slice = min(current_slice + 1, depth - 1)
			engine.set_current_slice(current_slice)
		elif glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
			current_slice = max(current_slice - 1, 0)
			engine.set_current_slice(current_slice)

		if not engine.want_capture_mouse():
			curr_x, curr_y = glfw.get_cursor_pos(window)
			if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
				
				if not first_mouse:
					dx = curr_x - last_x
					dy = curr_y - last_y
					engine.rotate_volume(dx, dy)
				first_mouse = False
    
			elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
				norm_x = curr_x / width
				norm_y = curr_y / height
				engine.set_lens_pos(norm_x, norm_y)
    
			else:
				first_mouse = True

			last_x, last_y = curr_x, curr_y

		import OpenGL.GL as gl
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		engine.update_uniforms(width, height)
		gl.glViewport(0, 0, width, height)
		engine.render_viewports(width, height)
		engine.render_ui()
		glfw.swap_buffers(window)

	glfw.terminate()

if __name__ == "__main__":
	main()