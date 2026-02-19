#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

using namespace std;

namespace py = pybind11;

class RadEngine {
private:
    GLuint volumeTexture = 0;
    int width = 0, height = 0, depth = 0;

public:
    RadEngine() {
        cout << "Engine Initialized." << endl;
    }

    void init_opengl() {
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            throw runtime_error("Failed to initialize GLAD");
        }
        cout << "Engine: OpenGL " << GLVersion.major << "." << GLVersion.minor << " ready." << endl;
    }

    void upload_volume(py::array_t<int16_t> numpy_volume) {
        py::buffer_info buf = numpy_volume.request();

        if (buf.ndim != 3) {
            throw runtime_error("Volume be a 3D array (Z, Y, X");
        }

        int depth = (int)buf.shape[0];
        int height = (int)buf.shape[1];
        int width = (int)buf.shape[2];

        if (volumeTexture == 0) {
            glGenTextures(1, &volumeTexture);
        }

        glBindTexture(GL_TEXTURE_3D, volumeTexture);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R16I, width, height, depth, 0,
            GL_RED_INTEGER, GL_SHORT, buf.ptr);

        cout << "Successfully uploaded " << width << "x" << height << "x" << depth
            << " volume to GPU Texture ID: " << volumeTexture << endl;
    }
};

PYBIND11_MODULE(radoptima_core, m) {
    py::class_<RadEngine>(m, "RadEngine")
        .def(py::init<>())
        .def("init_opengl", &RadEngine::init_opengl)
        .def("upload_volume", &RadEngine::upload_volume);

    m.def("check_cuda", []() {
        int count;
        cudaGetDeviceCount(&count);
        cout << "CUDA Devices: " << count << endl;
    });
}