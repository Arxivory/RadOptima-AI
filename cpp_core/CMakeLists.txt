get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(EXTERNAL_LIBS "${PROJECT_ROOT}/Libraries")

# FORCE the output to the root folder so Python finds it instantly
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_ROOT}")

# FORCE glad.c to be treated as C++ to match main.cu's linker
set_source_files_properties("${PROJECT_ROOT}/glad.c" PROPERTIES LANGUAGE CXX)


# Define the path
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/thirdparty/imgui")

# 1. Add headers to include path
include_directories(${IMGUI_DIR})

# 2. Collect the source files manually
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/imgui_impl_opengl3.cpp
)

set_source_files_properties(${IMGUI_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(radoptima_core SHARED 
    main.cu 
    "${PROJECT_ROOT}/glad.c"
    ${IMGUI_SOURCES}
)

# Standard Includes
target_include_directories(radoptima_core PRIVATE 
    "${PROJECT_ROOT}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
    "${EXTERNAL_LIBS}/include"
)

# Pybind11 / Python Logic
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
target_include_directories(radoptima_core PRIVATE "${PYBIND11_INCLUDE_DIR}" "${Python3_INCLUDE_DIRS}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; import os; print(os.path.join(sys.prefix, 'libs'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
link_directories("${PYTHON_LIB_DIR}")

# Linking
target_link_libraries(radoptima_core PRIVATE 
    "${EXTERNAL_LIBS}/lib/glfw3.lib"
    opengl32.lib
    user32.lib 
    gdi32.lib 
    shell32.lib
    "${Python3_LIBRARIES}"
)

set_target_properties(radoptima_core PROPERTIES 
    PREFIX "" 
    SUFFIX ".pyd"
    OUTPUT_NAME "radoptima_core"
)

# Critical for CUDA in VS2022
set_target_properties(radoptima_core PROPERTIES 
    CUDA_ARCHITECTURES 61
    LINKER_LANGUAGE CXX
)