cmake_minimum_required(VERSION 3.0.0)
project(cuda-raytracer)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(Qt5Gui REQUIRED)
#SET(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda-7.5")
find_package(CUDA REQUIRED)
find_package(PkgConfig)
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(GLEW QUIET REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(SYSTEM ${GLFW_INCLUDE_DIRS})
include_directories(SYSTEM "externlibs/")

link_libraries(${GLFW_LIBRARIES})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -Wall -Wextra -Wno-unused-function")
list(APPEND CUDA_NVCC_FLAGS "-Xptxas")
list(APPEND CUDA_NVCC_FLAGS " -O3")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=compute_20")

file(GLOB_RECURSE SRC_CUDA "src/*.cu")
file(GLOB_RECURSE UTIL_SOURCES "src/*.cpp")
file(GLOB_RECURSE UTIL_HEADERS "src/*.h")
include_directories("src")
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${UTIL_SOURCES} ${UTIL_HEADERS} ${SRC_CUDA})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${GLEW_LIBRARIES} GL GLU glfw assimp)
qt5_use_modules(${PROJECT_NAME} Gui)
