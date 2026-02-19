get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

add_library(radoptima_core SHARED 
    main.cu 
    "${PROJECT_ROOT}/glad.c"
)

set(EXTERNAL_LIBS "${PROJECT_ROOT}/Libraries")

target_include_directories(radoptima_core PRIVATE 
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
    "${EXTERNAL_LIBS}/include"
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_include_directories(radoptima_core PRIVATE "${PYBIND11_INCLUDE_DIR}")
target_include_directories(radoptima_core PRIVATE "${Python3_INCLUDE_DIRS}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; import os; print(os.path.join(sys.prefix, 'libs'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
link_directories("${PYTHON_LIB_DIR}")

target_link_libraries(radoptima_core PRIVATE 
    "${EXTERNAL_LIBS}/lib/glfw3.lib"
    opengl32.lib
    user32.lib 
    gdi32.lib 
    shell32.lib
    "${Python3_LIBRARIES}"
)

target_compile_options(radoptima_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/experimental:external /external:W0>)