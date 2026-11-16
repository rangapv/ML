cmake_minimum_required(VERSION 3.14)
project(ggml_tensors LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Locate the ggml source tree (for the headers) -------------------------
# Override with:  cmake -DGGML_DIR=/path/to/ggml ..
if(NOT GGML_DIR)
    foreach(candidate "/ggml" "$ENV{HOME}/ggml" "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
        if(EXISTS "${candidate}/include/ggml.h")
            set(GGML_DIR "${candidate}")
            break()
        endif()
    endforeach()
endif()
if(NOT GGML_DIR OR NOT EXISTS "${GGML_DIR}/include/ggml.h")
    message(FATAL_ERROR "ggml.h not found. Pass -DGGML_DIR=/path/to/ggml")
endif()

# --- Locate the already-built libraries ------------------------------------
# Defaults to GGML_DIR/build/src. Override with:  -DGGML_LIB_DIR=/path/to/libs
set(GGML_LIB_DIR "" CACHE PATH "Folder containing libggml*.so / *.dylib")
set(_hints "${GGML_LIB_DIR}" "${GGML_DIR}/build/src" "${GGML_DIR}/build/bin")

find_library(GGML_LIBRARY      NAMES ggml      HINTS ${_hints})
find_library(GGML_CPU_LIBRARY  NAMES ggml-cpu  HINTS ${_hints})
find_library(GGML_BASE_LIBRARY NAMES ggml-base HINTS ${_hints})

if(NOT GGML_LIBRARY)
    message(FATAL_ERROR
        "libggml not found in ${_hints}.\n"
        "Build ggml first, or pass -DGGML_LIB_DIR=/path/to/folder/with/libggml.so\n"
        "(if you changed paths, delete the build folder to clear CMake's cache)")
endif()
message(STATUS "ggml headers:   ${GGML_DIR}/include")
message(STATUS "ggml library:   ${GGML_LIBRARY}")

# --- The program -----------------------------------------------------------
add_executable(ggml_tensors ggml_tensors.c)
target_include_directories(ggml_tensors PRIVATE "${GGML_DIR}/include")

# Link whichever of the split libraries exist in your ggml version
target_link_libraries(ggml_tensors PRIVATE ${GGML_LIBRARY})
if(GGML_CPU_LIBRARY)
    target_link_libraries(ggml_tensors PRIVATE ${GGML_CPU_LIBRARY})
endif()
if(GGML_BASE_LIBRARY)
    target_link_libraries(ggml_tensors PRIVATE ${GGML_BASE_LIBRARY})
endif()
target_link_libraries(ggml_tensors PRIVATE m)

# Let the binary find the .so files at run time without LD_LIBRARY_PATH
get_filename_component(GGML_LIB_FOLDER "${GGML_LIBRARY}" DIRECTORY)
set_target_properties(ggml_tensors PROPERTIES BUILD_RPATH "${GGML_LIB_FOLDER}")
