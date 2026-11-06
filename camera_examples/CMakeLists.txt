cmake_minimum_required(VERSION 3.16)
project(mvs_camera_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

add_executable(mat_memory mat_memory.cpp)
target_include_directories(mat_memory PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mat_memory PRIVATE ${OpenCV_LIBS})

option(BUILD_MVS_CAPTURE "Build the MVS SDK C++ capture example" ON)
set(MVS_ROOT "/opt/MVS" CACHE PATH "MVS installation root")
set(MVS_LIBRARY_DIR "" CACHE PATH "Directory containing libMvCameraControl.so")

if(BUILD_MVS_CAPTURE)
    find_package(OpenCV 4 REQUIRED COMPONENTS imgcodecs)
    find_path(MVS_INCLUDE_DIR MvCameraControl.h HINTS "${MVS_ROOT}/include")
    find_library(
        MVS_LIBRARY
        NAMES MvCameraControl
        HINTS "${MVS_LIBRARY_DIR}" "${MVS_ROOT}/lib/64" "${MVS_ROOT}/lib/aarch64"
    )
    if(NOT MVS_INCLUDE_DIR OR NOT MVS_LIBRARY)
        message(FATAL_ERROR
            "Set MVS_ROOT and MVS_LIBRARY_DIR to the Linux SDK installation, "
            "or configure with -DBUILD_MVS_CAPTURE=OFF.")
    endif()
    add_executable(mvs_capture mvs_capture.cpp)
    target_include_directories(
        mvs_capture
        PRIVATE ${MVS_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(mvs_capture PRIVATE ${MVS_LIBRARY} ${OpenCV_LIBS})
    get_filename_component(MVS_RUNTIME_DIR "${MVS_LIBRARY}" DIRECTORY)
    set_target_properties(
        mvs_capture
        PROPERTIES BUILD_RPATH "${MVS_RUNTIME_DIR}"
    )
endif()
