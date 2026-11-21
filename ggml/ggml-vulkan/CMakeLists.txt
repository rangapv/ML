cmake_minimum_required(VERSION 3.14)
project(matmul_vk C CXX)
set(CMAKE_CXX_STANDARD 17)

# Point GGML_DIR at your ggml checkout: cmake -B build -DGGML_DIR=/path/to/ggml
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml" CACHE PATH "Path to ggml source")
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_DIR} ggml)

add_executable(matmul_vk matmul_vk.cpp)
target_link_libraries(matmul_vk PRIVATE ggml)
