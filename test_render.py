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
import threading
import tkinter as tk
from tkinter import filedialog

build_path = os.path.join(os.getcwd(), "out", "build", "x64-Debug", "cpp_core")

if os.path.exists(build_path):
    sys.path.append(build_path)
    print(f"Added build path: {build_path}")
else:
    print(f"Error: Could not find build folder at {build_path}")
    print("Make sure you have clicked 'Build All' in Visual Studio.")

STATE_EMPTY = 0
STATE_LOADING_RAW = 1
STATE_READY_RAW = 2
STATE_RUNNING_AI = 3
STATE_READY_ALL = 4

class AppState:
	def __init__(self):
		self.current_state = STATE_EMPTY
		self.status_message = "No volume loaded. Select a study directory"
		self.volume_raw = None
		self.volume_ai = None
		self.current_slice = 0
		self.depth = 0
		self.selected_path = ""

import radoptima_core

app_state = AppState()

last_x, last_y = 256, 256
first_mouse = True

def open_folder_dialog():
	root = tk.Tk()
	root.withdraw()
	root.attributes('-topmost', True)
	folder_selected = filedialog.askdirectory(initialdir="./data")
	root.destroy()
	return folder_selected

def bg_load_raw(directory, engine):
    global app_state
    try:
        explicit_vr_path = "./data/explicit_vr"
        app_state.current_state = STATE_LOADING_RAW
        app_state.status_message = "Clearing previous volume cache..."

        engine.reset_engine()

        temp_ai_dir = "./data/temp_ai_out"
        if os.path.exists(temp_ai_dir):
            shutil.rmtree(temp_ai_dir)

        app_state.volume_raw = None
        app_state.volume_ai = None

        app_state.status_message = "Parsing DICOM slices from disk..."
        volume, scale = load_dicom_volume(directory)

        app_state.volume_raw = volume
        app_state.depth = volume.shape[0]
        app_state.current_slice = app_state.depth // 2

        app_state.selected_path = directory
        app_state.status_message = f"Raw volume parsed ({app_state.depth} slices). Starting AI denoising..."

        if not has_ai_output(temp_ai_dir):
            convert_to_explicit_vr(directory, explicit_vr_path)

        print("No AI Output Found, Running AI")
        bg_run_ai(explicit_vr_path, engine, volume)

    except Exception as e:
        app_state.current_state = STATE_EMPTY
        app_state.status_message = f"Load Error: {str(e)}"

def bg_run_ai(original_path, engine, raw_volume=None):
    global app_state
    try:
        app_state.current_state = STATE_RUNNING_AI

        temp_ai_dir = "./data/temp_ai_out"
        if os.path.exists(temp_ai_dir):
            shutil.rmtree(temp_ai_dir)
        os.makedirs(temp_ai_dir)

        raw_file_count = len([
            f for f in os.listdir(original_path)
            if f.endswith('.DCM') or f.endswith('.dcm')
        ])

        app_state.status_message = f"RED-CNN denoising {raw_file_count} slices. GPU is focused, rendering paused..."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        denoise_dicom(
            dicom_path=original_path,
            savedir=temp_ai_dir,
            method=Methods.REDCNN,
            device=device
        )

        import time
        while True:
            ai_files = [
                f for f in os.listdir(temp_ai_dir)
                if f.endswith('.DCM') or f.endswith('.dcm')
            ]
            current_count = len(ai_files)
            app_state.status_message = f"Waiting for AI output... ({current_count}/{raw_file_count} slices written)"
            if current_count >= raw_file_count:
                break
            time.sleep(1.0)

        app_state.status_message = "Denoising complete. Loading AI volume..."
        ai_volume, _ = load_dicom_volume(temp_ai_dir)
        app_state.volume_ai = ai_volume

        if raw_volume is not None:
            engine.upload_volume(raw_volume)
            engine.set_current_slice(app_state.current_slice)

        engine.upload_ai_volume(ai_volume)

        app_state.current_state = STATE_READY_ALL
        app_state.status_message = f"Ready. {app_state.depth} slices | AI enhanced."

    except Exception as e:
        app_state.current_state = STATE_EMPTY
        app_state.status_message = f"AI Error: {str(e)}"

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
    global first_mouse, last_x, last_y, app_state

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

    tf_data = np.zeros((256, 4), dtype=np.float32)
    for i in range(256):
        intensity = i / 255.0
        tf_data[i] = [intensity, intensity * 0.5, 0.2, intensity]

    engine.update_transfer_function(tf_data)

    with open("shaders/raymarch.vert", "r") as f:
        v_src = f.read()
    with open("shaders/raymarch.frag", "r") as f:
        f_src = f.read()
    engine.compile_shader(v_src, f_src)

    engine.set_window_level(400, 40)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        width, height = glfw.get_framebuffer_size(window)

        engine.flush_pending_uploads()

        if app_state.current_state >= STATE_READY_RAW:
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                app_state.current_slice = min(app_state.current_slice + 1, app_state.depth - 1)
                engine.set_current_slice(app_state.current_slice)
            elif glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                app_state.current_slice = max(app_state.current_slice - 1, 0)
                engine.set_current_slice(app_state.current_slice)

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

        if app_state.current_state == STATE_READY_ALL:
            engine.update_uniforms(width, height)
            gl.glViewport(0, 0, width, height)
            engine.render_viewports(width, height)

        engine.render_ui(width, height)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
	main()