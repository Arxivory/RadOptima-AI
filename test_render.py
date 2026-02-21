import sys
import os
import glfw
import numpy as np
import ctypes
from data.dicom_loader import load_dicom_volume


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

def main():
	global first_mouse, last_x, last_y

	if not glfw.init():
		return

	glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
	glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
	glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

	window = glfw.create_window(800, 800, "RadOptima Engine Test", None, None)
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

	window_addr = ctypes.cast(window, ctypes.c_void_p).value
	# engine.init_imgui(window_addr)

	engine.setup_cube()

	data_path = "data/samples/chest"
	volume, volume_scale = load_dicom_volume(data_path)

	engine.upload_volume(volume)
	# engine.set_volume_scale(volume_scale[0], volume_scale[1], volume_scale[2])

	with open("shaders/raymarch.vert", "r") as f: v_src = f.read()
	with open("shaders/raymarch.frag", "r") as f: f_src = f.read()
	engine.compile_shader(v_src, f_src)

	engine.set_window_level(400, 40)

	# engine.render_ui()

	while not glfw.window_should_close(window):
		glfw.poll_events()

		if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
			curr_x, curr_y = glfw.get_cursor_pos(window)
			if not first_mouse:
				dx = curr_x - last_x
				dy = curr_y - last_y
				engine.rotate_volume(dx, dy)
			last_x, last_y = curr_x, curr_y
			first_mouse = False
		else:
			first_mouse = True

		import OpenGL.GL as gl
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		engine.update_uniforms()
		engine.render()
		glfw.swap_buffers(window)

	glfw.terminate()

if __name__ == "__main__":
	main()