#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace glm;

namespace py = pybind11;

class RadEngine {
private:
    GLuint volumeTexture = 0, shaderProgram;
    int width = 0, height = 0, depth = 0;
    float windowWidth = 400, windowLevel = 40;
    mat4 modelMatrix = mat4(1.0f);
    vec3 cameraPos = vec3(0.5f, 0.5f, 2.0f);
    unsigned int cubeVAO, cubeVBO, cubeEBO;

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

    void setup_cube() {
        float vertices[] = {
        0,0,0,  1,0,0,  1,1,0,  0,1,0, 
        0,0,1,  1,0,1,  1,1,1,  0,1,1
        };

        unsigned int indices[] = {
            0,1,2, 2,3,0,
            1,5,6, 6,2,1,
            7,6,5, 5,4,7,
            4,0,3, 3,7,4,
            4,5,1, 1,0,4,
            3,2,6, 6,7,3
        };

        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        glGenBuffers(1, &cubeEBO);

        glBindVertexArray(cubeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }

    void set_window_level(float width, float level) {
        windowWidth = width;
        windowLevel = level;
    }

    void update_shader_params() {
        glUseProgram(shaderProgram);
        glUniform1f(glGetUniformLocation(shaderProgram, "windowWidth"), windowWidth);
        glUniform1f(glGetUniformLocation(shaderProgram, "windowLevel"), windowLevel);

        unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, value_ptr(modelMatrix));
    }

    void set_model_matrix(py::array_t<float> matrix) {
        auto r = matrix.unchecked<2>();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                modelMatrix[i][j] = r(i, j);
    }

    void rotate_volume(float deltaX, float deltaY) {
        float sensitivity = 0.01f;

        modelMatrix = rotate(modelMatrix, deltaX * sensitivity, glm::vec3(0, 1, 0));
        modelMatrix = rotate(modelMatrix, deltaY * sensitivity, glm::vec3(1, 0, 0));
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

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R16I, width, height, depth, 0,
            GL_RED_INTEGER, GL_SHORT, buf.ptr);

        cout << "Successfully uploaded " << width << "x" << height << "x" << depth
            << " volume to GPU Texture ID: " << volumeTexture << endl;
    }

    void update_uniforms() {
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &modelMatrix[0][0]);

        glm::mat4 invModel = glm::inverse(modelMatrix);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "invModel"), 1, GL_FALSE, &invModel[0][0]);

        glUniform3fv(glGetUniformLocation(shaderProgram, "worldEyePos"), 1, &cameraPos[0]);
    }

    void compile_shader(const char* vertexSource, const char* fragmentSource) {
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        
        glLinkProgram(shaderProgram);

        glUseProgram(shaderProgram);

        cout << "Shader Program is Compiled and Active." << endl;
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