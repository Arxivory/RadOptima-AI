#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void check_cuda() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
    }
    else {
        std::cout << "RadOptima Engine: Found " << deviceCount << " GPU(s)." << std::endl;
    }
}

PYBIND11_MODULE(radoptima_core, m) {
    m.doc() = "RadOptima AI CUDA-OpenGL Engine";
    m.def("check_cuda", &check_cuda, "A function to check GPU availability");
}