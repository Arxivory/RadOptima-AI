# RadOptima AI

**RadOptima AI** is a professional-grade medical imaging platform designed to bridge the gap between heavy-duty C++ engineering and modern Deep Learning. 

It features a high-speed **CUDA-accelerated 3D Volume Renderer** and a **Python-based AI pipeline** for real-time CT scan enhancement.



---

## Key Features
* **Hybrid Engine:** C++ backend for performance, Python frontend for flexibility.
* **Direct GPU Interop:** Zero-copy data transfer between PyTorch (AI) and OpenGL (Rendering) using CUDA.
* **DICOM Intelligence:** Native support for Hounsfield Unit (HU) windowing and medical metadata parsing.
* **AI Denoising:** Real-time 3D U-Net implementation for low-dose CT enhancement.

## Technical Architecture
RadOptima AI utilizes a **Dual-Stage Architecture**:
1.  **Core Engine (C++/CUDA):** Handles the mathematical heavy lifting of Raymarching and memory management for 3D textures.
2.  **Application Layer (Python):** Manages the UI (ImGui), DICOM loading, and the PyTorch inference model.

## Tech Stack
* **Language:** Python 3.11.9, C++ 17
* **Graphics:** OpenGL (GLSL Shaders)
* **Compute:** CUDA 12.4 (Optimized for Pascal/10-series architectures)
* **AI:** PyTorch, NumPy
* **Bridge:** PyBind11