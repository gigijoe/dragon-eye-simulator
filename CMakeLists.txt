cmake_minimum_required(VERSION 2.8)
project( dragon-eye )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_definitions(-std=c++11)

INCLUDE_DIRECTORIES(/usr/local/include/opencv4)

LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)

cuda_add_executable( dragon-eye-cuda dragon-eye-cuda.cpp)
set(dragon-eye-cuda dragon-eye-cuda PROPERTY CUDA_STANDARD 11)
target_link_libraries( dragon-eye-cuda ${OpenCV_LIBS} Threads::Threads)

