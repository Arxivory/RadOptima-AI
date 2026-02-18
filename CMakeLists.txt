cmake_minimum_required(VERSION 3.18)
project(RadOptimaAI LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(cpp_core)