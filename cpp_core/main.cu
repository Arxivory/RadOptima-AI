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

    void upload_volume(py::array_t<int16_t> numpy_volume) {
        py::buffer_info buf = numpy_volume.request();

        if (buf.ndim != 3) {
            throw runtime_error("Volume be a 3D array (Z, Y, X");
        }

        int depth = (int)buf.shape[0];
        int height = (int)buf.shape[1];
        int width = (int)buf.shape[2];

        cout << "C++ Received Volume: " << width << "x" << height << "x" << depth << endl;

        int16_t* ptr = static_cast<int16_t*>(buf.ptr);

        cout << "First voxel value: " << ptr[0] << " HU" << endl;
    
    }
};

PYBIND11_MODULE(radoptima_core, m) {
    py::class_<RadEngine>(m, "RadEngine")
        .def(py::init<>())
        .def("upload_volume", &RadEngine::upload_volume);

    m.def("check_cuda", []() {
        int count;
        cudaGetDeviceCount(&count);
        cout << "CUDA Devices: " << count << endl;
    });
}