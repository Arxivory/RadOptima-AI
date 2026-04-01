#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <windows.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <cstdint>
#include "imgui_impl_win32.h"

using namespace std;
using namespace glm;
using namespace ImGui;

namespace py = pybind11;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static WNDPROC g_GlfwWndProc = NULL;

LRESULT CALLBACK MyWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, uMsg, wParam, lParam))
        return true;

    return CallWindowProc(g_GlfwWndProc, hWnd, uMsg, wParam, lParam);
}

class RadEngine {
private:
    GLuint volumeTexture = 0, tfTexture = 0, shaderProgram;
    int width = 0, height = 0, depth = 0;
    float windowWidth = 400, windowLevel = 40;
    mat4 modelMatrix = mat4(1.0f);
    vec3 cameraPos = vec3(0.5f, 0.5f, 2.0f);
    unsigned int cubeVAO, cubeVBO, cubeEBO;
    float tf_opacity_power = 2.0f, tf_multiplier = 0.05, stepSize = 0.002f;

	vec3 lensCenter = vec3(0.5f, 0.5f, 0.5f);
    float lensRadius = 0.2f;
    bool lensEnabled = true;

public:
    RadEngine() {
        cout << "Engine Initialized." << endl;
    }

    void init_opengl() {
        std::cout << "[C++] Bypassing GLFW to find OpenGL Context..." << std::endl;

        HMODULE libGL = GetModuleHandleA("opengl32.dll");
        if (!libGL) {
            throw std::runtime_error("Could not find opengl32.dll. Is a GPU driver installed?");
        }

        auto wgl_get_proc_address = (void* (WINAPI*)(const char*))GetProcAddress(libGL, "wglGetProcAddress");

        if (!gladLoadGLLoader((GLADloadproc)wgl_get_proc_address)) {
            if (!gladLoadGL()) {
                throw std::runtime_error("GLAD Fail: No current OpenGL context found. Make sure Python called make_context_current.");
            }
        }

        if (glGenTextures == nullptr) {
            throw std::runtime_error("GLAD loaded but function pointers are NULL. Context mismatch.");
        }

        std::cout << "SUCCESS: OpenGL Context captured from Python!" << std::endl;
        std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;
    }

    void init_imgui(uintptr_t hwnd_ptr) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        HWND hwnd = (HWND)hwnd_ptr;
        if (!hwnd) throw std::runtime_error("Received null HWND.");

        ImGui_ImplWin32_Init(hwnd);
        ImGui_ImplOpenGL3_Init("#version 450");

        g_GlfwWndProc = (WNDPROC)SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)MyWindowProc);
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

    void update_lens_uniform() {
        glUseProgram(shaderProgram);
        glUniform3fv(glGetUniformLocation(shaderProgram, "lensCenter"), 1, value_ptr(lensCenter));
        glUniform1f(glGetUniformLocation(shaderProgram, "lensRadius"), lensRadius);
		glUniform1i(glGetUniformLocation(shaderProgram, "lensEnabled"), lensEnabled ? 1 : 0);
    }

    void update_shader_params() {
        glUseProgram(shaderProgram);
        glUniform1f(glGetUniformLocation(shaderProgram, "windowWidth"), windowWidth);
        glUniform1f(glGetUniformLocation(shaderProgram, "windowLevel"), windowLevel);

        unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, value_ptr(modelMatrix));

        glUniform1f(glGetUniformLocation(shaderProgram, "windowWidth"), windowWidth);
        glUniform1f(glGetUniformLocation(shaderProgram, "windowLevel"), windowLevel);
        glUniform1f(glGetUniformLocation(shaderProgram, "tf_opacity_power"), tf_opacity_power);
        glUniform1f(glGetUniformLocation(shaderProgram, "tf_multiplier"), tf_multiplier);
        glUniform1f(glGetUniformLocation(shaderProgram, "stepSize"), stepSize);
    }

    void set_model_matrix(py::array_t<float> matrix) {
        auto r = matrix.unchecked<2>();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                modelMatrix[i][j] = r(i, j);
    }

    void rotate_volume(float deltaX, float deltaY) {
        /*ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) {
            return;
        }*/

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

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

        float borderColor[] = { -1024.0f, 0.0f, 0.0f, 0.0f };
        glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, borderColor);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R16I, width, height, depth, 0,
            GL_RED_INTEGER, GL_SHORT, buf.ptr);
    }

    void update_transfer_function(py::array_t<float> lut) {
        py::buffer_info buf = lut.request();
        if (tfTexture == 0) glGenTextures(1, &tfTexture);

        glBindTexture(GL_TEXTURE_1D, tfTexture);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, (GLsizei)buf.shape[0], 0, GL_RGBA, GL_FLOAT, buf.ptr);

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }

    void update_uniforms(int winW, int winH) {
        float aspect = (float)winW / (float)winH;
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &modelMatrix[0][0]);

        glm::mat4 invModel = glm::inverse(modelMatrix);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "invModel"), 1, GL_FALSE, &invModel[0][0]);

        glUniform3fv(glGetUniformLocation(shaderProgram, "eyePos"), 1, &cameraPos[0]);

        mat4 projection = perspective(radians(45.0f), aspect, 0.1f, 100.0f);
        mat4 view = lookAt(cameraPos, vec3(0.5f, 0.5f, 0.5f), vec3(0, 1, 0));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);

        update_lens_uniform();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, volumeTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "volumeTexture"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tfTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "transferFunction"), 1);

        glUniform1f(glGetUniformLocation(shaderProgram, "tf_multiplier"), tf_multiplier);
        glUniform1f(glGetUniformLocation(shaderProgram, "stepSize"), stepSize);
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

    void set_volume_scale(float sx, float sy, float sz) {
        modelMatrix = scale(modelMatrix, vec3(sx, sy, sz));

        cout << "Volume Scale Applied: " << sx << ", " << sy << ", " << sz << endl;
    }

    void render() {
        glUseProgram(shaderProgram);
        glBindVertexArray(cubeVAO);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }

    void render_ui() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplWin32_NewFrame();
        NewFrame();

        Begin("RadOptima Controls");
        SliderFloat("Window Width", &windowWidth, 1.0f, 2000.0f);
        SliderFloat("Window Level", &windowLevel, -1000.0f, 1000.0f);
        SliderFloat("Opacity Power", &tf_opacity_power, 1.0f, 10.0f);
        SliderFloat("Opacity Multiplier", &tf_multiplier, 0.001f, 0.2f);
        SliderFloat("Step Size", &stepSize, 0.0005f, 0.01f);
        End();

		Begin("Lens Controls");
		SliderFloat3("Lens Center", &lensCenter[0], 0.0f, 1.0f);
		SliderFloat("Lens Radius", &lensRadius, 0.01f, 0.5f);
		Checkbox("Enable Lens", &lensEnabled);
		End();

        Render();
        ImGui_ImplOpenGL3_RenderDrawData(GetDrawData());
    }
};

PYBIND11_MODULE(radoptima_core, m) {
    py::class_<RadEngine>(m, "RadEngine")
        .def(py::init<>())
        .def("init_opengl", &RadEngine::init_opengl)
        .def("setup_cube", &RadEngine::setup_cube)
        .def("upload_volume", &RadEngine::upload_volume)
        .def("set_window_level", &RadEngine::set_window_level)
        .def("rotate_volume", &RadEngine::rotate_volume)
        .def("update_uniforms", &RadEngine::update_uniforms)
        .def("compile_shader", &RadEngine::compile_shader)
        .def("set_volume_scale", &RadEngine::set_volume_scale)
        .def("render", &RadEngine::render)
        .def("init_imgui", &RadEngine::init_imgui)
        .def("render_ui", &RadEngine::render_ui)
        .def("update_transfer_function", &RadEngine::update_transfer_function)
        .def("want_capture_mouse", [](RadEngine& self) {
		    return ImGui::GetIO().WantCaptureMouse;
        });
}