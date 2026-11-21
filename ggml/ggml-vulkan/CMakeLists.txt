cmake_minimum_required(VERSION 3.14)
project(matmul_vk C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(ggml REQUIRED)

add_executable(matmul_vk matmul_vk.cpp)
target_link_libraries(matmul_vk PRIVATE ggml::ggml)
