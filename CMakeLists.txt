cmake_minimum_required(VERSION 3.23)
project(GPURecursiveMP CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

add_executable(GPURecursiveMP main.cu)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(GPURecursiveMP -lcurand -lcublasLt -lcublas -lblas)

target_link_libraries(GPURecursiveMP ${OpenCV_LIBS})

set_target_properties(GPURecursiveMP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
