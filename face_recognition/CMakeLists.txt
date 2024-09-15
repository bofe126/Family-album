cmake_minimum_required(VERSION 3.10)
project(face_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 OpenCV 路径，如果需要的话请修改这个路径
set(OpenCV_DIR "C:/Program Files/opencv/build")
find_package(OpenCV REQUIRED dnn)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR to the correct path.")
endif()

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/packages/Microsoft.ML.OnnxRuntime.1.12.0")

# 使用 find_package 查找 ONNX Runtime
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(onnxruntime REQUIRED)

if(MSVC)
  add_compile_options("/utf-8")
endif()

# 添加源文件
set(SOURCES
    face_recognition.cpp
)

add_library(face_recognition SHARED ${SOURCES})

# 设置包含目录
target_include_directories(face_recognition PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/build/native/include
)

# 设置链接目录
target_link_directories(face_recognition PRIVATE
    ${ONNXRUNTIME_ROOT}/runtimes/win-x64/native
)

# 链接库
target_link_libraries(face_recognition 
    ${OpenCV_LIBS}
    onnxruntime
)

# 定义模型文件的源路径和目标路径
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
set(YOLOV5_MODEL_SOURCE "${ASSETS_DIR}/yolov5m.onnx")
set(ARCFACE_MODEL_SOURCE "${ASSETS_DIR}/arcface_model.onnx")
set(YOLOV5_MODEL_DEST "$<TARGET_FILE_DIR:face_recognition>/yolov5m.onnx")
set(ARCFACE_MODEL_DEST "$<TARGET_FILE_DIR:face_recognition>/arcface_model.onnx")

# 检查源文件是否存在
if(NOT EXISTS ${YOLOV5_MODEL_SOURCE})
    message(FATAL_ERROR "Source YOLOV5 model file not found: ${YOLOV5_MODEL_SOURCE}")
endif()
if(NOT EXISTS ${ARCFACE_MODEL_SOURCE})
    message(FATAL_ERROR "Source ArcFace model file not found: ${ARCFACE_MODEL_SOURCE}")
endif()

# 添加自定义命令来复制文件
add_custom_command(
    TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:face_recognition>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${YOLOV5_MODEL_SOURCE}" "${YOLOV5_MODEL_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ARCFACE_MODEL_SOURCE}" "${ARCFACE_MODEL_DEST}"
    COMMENT "Copying model files to output directory"
)

# 复制生成的 DLL 文件到 assets 目录
add_custom_command(
    TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:face_recognition>"
        "${ASSETS_DIR}/face_recognition.dll"
    COMMENT "Copying generated face_recognition.dll to assets directory"
)

# 复制 ONNX Runtime DLL 到目标目录
add_custom_command(
    TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.dll"
        "$<TARGET_FILE_DIR:face_recognition>"
    COMMENT "Copying onnxruntime.dll to target directory"
)

# 打印一些调试信息
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")
message(STATUS "YOLOV5_MODEL_SOURCE: ${YOLOV5_MODEL_SOURCE}")
message(STATUS "ARCFACE_MODEL_SOURCE: ${ARCFACE_MODEL_SOURCE}")
message(STATUS "DLL_DEST: ${ASSETS_DIR}/face_recognition.dll")
