add_library(radoptima_core SHARED main.cu)

set_target_properties(radoptima_core PROPERTIES PREFIX "" SUFFIX ".pyd")

target_include_directories(radoptima_core PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include")

find_package(OpenGL REQUIRED)
target_link_libraries(radoptima_core PRIVATE OpenGL::GL)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find pybind11 include directory. Did you run 'pip install pybind11'?")
endif()

target_include_directories(radoptima_core PRIVATE "${PYBIND11_INCLUDE_DIR}")
target_include_directories(radoptima_core PRIVATE "${Python3_INCLUDE_DIRS}")

# 1. Find the Python Library path automatically
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; import os; print(os.path.join(sys.prefix, 'libs'))"
    OUTPUT_VARIABLE PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 2. Tell the Linker where to look for .lib files
link_directories("${PYTHON_LIB_DIR}")

# 3. Explicitly link the Python library to your target
target_link_libraries(radoptima_core PRIVATE "${Python3_LIBRARIES}")

target_compile_options(radoptima_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/experimental:external /external:W0>)