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
#include <vector>

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

struct ProbeData {
    int16_t rawHU;
    int16_t aiHU;
    float x_norm, y_norm;
    bool isActive;
};

struct ROIResult {
    float mean;
    float stdDev;
};

class RadEngine {
private:
    GLuint volumeTexture = 0, tfTexture = 0, shaderProgram = 0, volumeTextureAI = 0;
    int width = 0, height = 0, depth = 0;
    float windowWidth = 400, windowLevel = 40;
    mat4 modelMatrix = mat4(1.0f);
    vec3 cameraPos = vec3(0.5f, 0.5f, 2.0f);
    unsigned int cubeVAO, cubeVBO, cubeEBO;
    float tf_opacity_power = 2.0f, tf_multiplier = 0.05, stepSize = 0.002f;

	vec3 lensCenter = vec3(0.5f, 0.5f, 0.5f);
    float lensRadius = 0.2f;
    bool lensEnabled = true;

    bool diffMode = false;

    int currentSlice = 0;
    int compareMode2D = 0;
    float sliderX = 0.5;

    std::vector<int16_t> cpuVolumeRaw;
    std::vector<int16_t> cpuVolumeAI;
    ProbeData currentProbe;

    ROIResult cachedRawStats = { 0,0 };
    ROIResult cachedAIStats = { 0,0 };
    vec3 lastLensPos = vec3(-1.0f);
    float lastLensRadius = -1.0f;
    int lastSlice = -1;

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

    void set_lens_pos(float x, float y) {
        lensCenter.x = x;
        lensCenter.y = 1.0f - y;
    }

    void set_diff_mode(bool enabled) {
        diffMode = enabled;
    }

    void set_current_slice(int s) {
        currentSlice = s;
    }

    void render_viewports(int winW, int winH) {
        float splitAspect = (float)(winW / 2) / (float)winH;
        glViewport(0, 0, winW / 2, winH);

        mat4 projection = perspective(radians(45.0f), splitAspect, 0.1f, 100.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, value_ptr(projection));

        glUniform1i(glGetUniformLocation(shaderProgram, "is2DView"), 0);
        render();

        glViewport(winW / 2, 0, winW / 2, winH);
        glUniform1i(glGetUniformLocation(shaderProgram, "is2DView"), 1);

        mat4 identity = mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, value_ptr(identity));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, value_ptr(identity));

        mat4 orthoProj = ortho(0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, value_ptr(orthoProj));

        glUniform1f(glGetUniformLocation(shaderProgram, "sliceZ"), (float)currentSlice / (float)depth);
        render();
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
        float sensitivity = 0.01f;

        modelMatrix = rotate(modelMatrix, deltaX * sensitivity, glm::vec3(0, 1, 0));
        modelMatrix = rotate(modelMatrix, deltaY * sensitivity, glm::vec3(1, 0, 0));
    }

    void upload_ai_volume(py::array_t<int16_t> numpy_volume) {
        py::buffer_info buf = numpy_volume.request();
        this->depth = (int)buf.shape[0];
        this->height = (int)buf.shape[1];
        this->width = (int)buf.shape[2];

        int16_t* ptr = static_cast<int16_t*>(buf.ptr);
        cpuVolumeAI.assign(ptr, ptr + (depth * height * width));

        if (volumeTextureAI == 0)
            glGenTextures(1, &volumeTextureAI);

		glBindTexture(GL_TEXTURE_3D, volumeTextureAI);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, width, height, depth, 0,
            GL_RED, GL_SHORT, buf.ptr);
    }

    void upload_volume(py::array_t<int16_t> numpy_volume) {
        py::buffer_info buf = numpy_volume.request();

        if (buf.ndim != 3) {
            throw runtime_error("Volume be a 3D array (Z, Y, X");
        }

        this->depth = (int)buf.shape[0];
        this->height = (int)buf.shape[1];
        this->width = (int)buf.shape[2];

        int16_t* ptr = static_cast<int16_t*>(buf.ptr);
        cpuVolumeRaw.assign(ptr, ptr + (depth * height * width));

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

        glUniform1i(glGetUniformLocation(shaderProgram, "diffMode"), diffMode ? 1 : 0);
		glUniform1i(glGetUniformLocation(shaderProgram, "compareMode2D"), compareMode2D);
        glUniform1f(glGetUniformLocation(shaderProgram, "sliderX"), sliderX);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, volumeTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "volumeTexture"), 0);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_3D, volumeTextureAI);
        glUniform1i(glGetUniformLocation(shaderProgram, "volumeTextureAI"), 2);

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

    void update_probe(int winW, int winH) {
        ImGuiIO& io = ImGui::GetIO();
        currentProbe.isActive = false;

        float xMin = winW / 2.0f;
        float xMax = (float)winW;

        if (io.MousePos.x >= xMin && io.MousePos.x <= xMax &&
            io.MousePos.y >= 0 && io.MousePos.y <= winH) {

            float u = (io.MousePos.x - xMin) / (winW / 2.0f);
            float v = io.MousePos.y / (float)winH;

            int vx = glm::clamp((int)(u * width), 0, width - 1);
            int vy = glm::clamp((int)(v * height), 0, height - 1);
            int vz = currentSlice;

            size_t idx = (size_t)vz * (width * height) + (vy * width) + vx;

            if (idx < cpuVolumeRaw.size()) {
                currentProbe.rawHU = cpuVolumeRaw[idx];
                currentProbe.aiHU = !cpuVolumeAI.empty() ? cpuVolumeAI[idx] : 0;
                currentProbe.x_norm = u;
                currentProbe.y_norm = v;
                currentProbe.isActive = true;
            }
        }
    }

    void DrawHistogram() {
        if (cpuVolumeRaw.empty()) return;

        static float histData[256];
        std::fill(std::begin(histData), std::end(histData), 0.0f);

        size_t sliceSize = width * height;
        size_t startIdx = (size_t)currentSlice * sliceSize;

        float minHU = windowLevel - (windowWidth / 2.0f);
        float maxHU = windowLevel + (windowWidth / 2.0f);

        for (size_t i = 0; i < sliceSize; i++) {
            int16_t val = cpuVolumeRaw[startIdx + i];
            int bin = (int)(((float)val - minHU) / windowWidth * 255.0f);
            if (bin >= 0 && bin < 256) histData[bin]++;
        }

        ImGui::Begin("Live Analytics");
        ImGui::Text("Slice Intensity Distribution");
        ImGui::PlotHistogram("##HUHist", histData, 256, 0, NULL, 0.0f, sliceSize * 0.05f, ImVec2(0, 80));
        ImGui::TextDisabled("Air (-1000) <---> Bone (+1000)");
        ImGui::End();
    }

    ROIResult calculateROI(const std::vector<int16_t>& buffer) {
        if (buffer.empty()) return { 0, 0 };

        float sum = 0, sq_sum = 0;
        int count = 0;

        int minX = glm::clamp((int)((lensCenter.x - lensRadius) * width), 0, width - 1);
        int maxX = glm::clamp((int)((lensCenter.x + lensRadius) * width), 0, width - 1);
        int minY = glm::clamp((int)((1.0f - (lensCenter.y + lensRadius)) * height), 0, height - 1);
        int maxY = glm::clamp((int)((1.0f - (lensCenter.y - lensRadius)) * height), 0, height - 1);

        float r2 = lensRadius * lensRadius; 

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                float u = (float)x / width;
                float v = (float)y / height;

                float dx = u - lensCenter.x;
                float dy = (1.0f - v) - lensCenter.y;
                float distSq = dx * dx + dy * dy;

                if (distSq < r2) {
                    int16_t val = buffer[(size_t)currentSlice * width * height + y * width + x];
                    sum += val;
                    sq_sum += val * val;
                    count++;
                }
            }
        }

        if (count == 0) return { 0, 0 };
        float mean = sum / count;
        float variance = (sq_sum / count) - (mean * mean);
        return { mean, sqrt(max(0.0f, variance)) };
    }

    void DrawClinicalAnalytics() {
        Begin("Clinical Analytics");

        if (lensEnabled) {
            bool isDirty = (lensCenter != lastLensPos ||
                lensRadius != lastLensRadius ||
                currentSlice != lastSlice);

            if (isDirty) {
                cachedRawStats = calculateROI(cpuVolumeRaw);
                cachedAIStats = !cpuVolumeAI.empty() ? calculateROI(cpuVolumeAI) : ROIResult{ 0,0 };

                lastLensPos = lensCenter;
                lastLensRadius = lensRadius;
                lastSlice = currentSlice;
            }

            if (BeginTable("StatsTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                TableSetupColumn("Metric");
                TableSetupColumn("Raw");
                TableSetupColumn("AI Denoised");
                TableHeadersRow();

                TableNextRow();
                TableSetColumnIndex(0); Text("Mean Intensity");
                TableSetColumnIndex(1); Text("%.1f HU", cachedRawStats.mean);
                TableSetColumnIndex(2); Text("%.1f HU", cachedAIStats.mean);

                TableNextRow();
                TableSetColumnIndex(0); Text("Noise (StdDev)");
                TableSetColumnIndex(1); Text("%.2f", cachedRawStats.stdDev);
                TableSetColumnIndex(2); TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.2f", cachedAIStats.stdDev);

                EndTable();
            }

            if (cachedRawStats.stdDev > 0.001f) {
                float reduction = (1.0f - (cachedAIStats.stdDev / cachedRawStats.stdDev)) * 100.0f;
                Separator();
                Text("Noise Reduction:"); SameLine();
                TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%.1f%%", reduction);

                PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                ProgressBar(reduction / 100.0f, ImVec2(-1, 15), "");
                PopStyleColor();
            }
        }
        else {
            TextDisabled("Enable Lens to see ROI Statistics.");
        }

        End();
    }

    void render() {
        glUseProgram(shaderProgram);
        glBindVertexArray(cubeVAO);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }

    void render_ui(int winW, int winH) {
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

        Begin("Clinical Presets");
        if (Button("Bone (400/1800)")) { windowLevel = 400; windowWidth = 1800; }
        SameLine();
        if (Button("Lung (-600/1500)")) { windowLevel = -600; windowWidth = 1500; }

        if (Button("Soft Tissue (40/400)")) { windowLevel = 40; windowWidth = 400; }
        SameLine();
        if (Button("Mediastinum (40/350)")) { windowLevel = 40; windowWidth = 350; }

        Separator();
        Checkbox("AI Difference Mode", &diffMode);
        TextDisabled("Shows (Raw - AI) to visualize noise removal.");
        End();

		Begin("Lens Controls");
		SliderFloat3("Lens Center", &lensCenter[0], 0.0f, 1.0f);
		SliderFloat("Lens Radius", &lensRadius, 0.01f, 0.5f);
		Checkbox("Enable Lens", &lensEnabled);
		End();

        Begin("Slice Viewer");
        if (SliderInt("Current Slice", &currentSlice, 0, depth - 1)) {
        }
        Text("Total Slices: %d", depth);

        Separator();

        Text("Intensity Range:");
        float low = windowLevel - (windowWidth / 2.0f);
        float high = windowLevel + (windowWidth / 2.0f);
        Text("%.0f HU <---> %.0f HU", low, high);
        End();

        Begin("Slice Comparison");

        if (RadioButton("Lens", compareMode2D == 1))
            compareMode2D = 1;

        SameLine();
        if (RadioButton("Slider", compareMode2D == 2))
            compareMode2D = 2;

        SameLine();
        if (RadioButton("Raw", compareMode2D == 0))
            compareMode2D = 0;

        if (compareMode2D == 2) {
            SliderFloat("Slice Split", &sliderX, 0.0f, 1.0f);
        }

        End();

        update_probe(winW, winH);

        if (currentProbe.isActive && !ImGui::GetIO().WantCaptureMouse) {
            ImGui::SetNextWindowBgAlpha(0.7f);
            ImGui::BeginTooltip();

            ImGui::TextDisabled("Coordinate: %.2f, %.2f (Slice %d)",
                currentProbe.x_norm, currentProbe.y_norm, currentSlice);
            ImGui::Separator();

            ImGui::Text("Raw: %d HU", currentProbe.rawHU);

            if (!cpuVolumeAI.empty()) {
                ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "AI:  %d HU", currentProbe.aiHU);

                float diff = (float)abs(currentProbe.rawHU - currentProbe.aiHU);
                ImGui::TextDisabled("Noise Delta: %.1f", diff);
            }

            ImGui::EndTooltip();
        }

		DrawHistogram();

        DrawClinicalAnalytics();

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
        .def("upload_ai_volume", &RadEngine::upload_ai_volume)
        .def("set_lens_pos", &RadEngine::set_lens_pos)
        .def("set_diff_mode", &RadEngine::set_diff_mode)
		.def("set_current_slice", &RadEngine::set_current_slice)
		.def("render_viewports", &RadEngine::render_viewports)
        .def("want_capture_mouse", [](RadEngine& self) {
		    return ImGui::GetIO().WantCaptureMouse;
        });
}