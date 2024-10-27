if(MSVC)
    add_compile_options(/utf-8)
endif()

cmake_minimum_required(VERSION 3.10)
project(face_recognition)
set(CMAKE_SYSTEM_VERSION 10.0)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 OpenCV 路径
set(OpenCV_DIR "C:/Program Files/opencv/build")
set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
set(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-win-x64-1.19.2")

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR to the directory containing OpenCVConfig.cmake")
endif()

# 添加源文件
set(SOURCES
    src/face_recognition.cpp
    src/face_detector.cpp
    src/arc_face_extractor.cpp
    src/utils.cpp
    src/globals.cpp
    src/model_cache.cpp
    src/performance.cpp
)

add_library(face_recognition SHARED ${SOURCES})

# 包含目录
target_include_directories(face_recognition
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        "${OpenCV_DIR}/include"
        "${OpenCV_DIR}/include/opencv2"
        ${ONNXRUNTIME_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 添加编译选项
if(MSVC)
    target_compile_options(face_recognition PRIVATE 
        /utf-8
        /W4
        /wd4819  # 禁用代码页警告
        /wd4244  # 禁用类型转换警告
        /wd4267  # 禁用 size_t 到 int 的转换警告
        /wd4458  # 禁用参数隐藏类成员的警告
    )
endif()

# 添加预编译头支持
if(MSVC)
    target_precompile_headers(face_recognition PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/pch.h
    )
endif()

# 链接库
target_link_libraries(face_recognition
    PRIVATE
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# 添加 OpenCV 的具体库文件路径
target_link_directories(face_recognition PRIVATE
    "${OpenCV_DIR}/x64/vc16/lib"
)

# 设置输出目录
set_target_properties(face_recognition PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# 复制依赖 DLL
add_custom_command(TARGET face_recognition POST_BUILD
    # 复制 ONNX Runtime DLL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:face_recognition>"
    # 复制所有 OpenCV DLL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OpenCV_BIN_DIR}"
        "$<TARGET_FILE_DIR:face_recognition>"
    # 复制生成的 DLL 到 assets 目录
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:face_recognition>"
        "${CMAKE_CURRENT_SOURCE_DIR}/../assets/face_recognition.dll"
)

# 添加调试信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
