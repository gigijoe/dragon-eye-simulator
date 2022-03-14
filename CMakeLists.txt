cmake_minimum_required(VERSION 2.8)
project( dragon-eye-simulator )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_definitions(-std=c++11)

INCLUDE_DIRECTORIES(/usr/local/include/opencv4)

LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)

# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/usr/local/TensorRT-7.2.3.4/include/)
link_directories(/usr/local/TensorRT-7.2.3.4/lib/)

cuda_add_executable( dragon-eye-simulator dragon-eye-simulator.cpp)
set(dragon-eye-simulator dragon-eye-simulator PROPERTY CUDA_STANDARD 11)
target_link_libraries( dragon-eye-simulator ${OpenCV_LIBS} Threads::Threads)

target_link_libraries(dragon-eye-simulator nvinfer)
target_link_libraries(dragon-eye-simulator cudart)

