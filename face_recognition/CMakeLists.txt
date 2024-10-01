cmake_minimum_required(VERSION 3.10)
project(face_recognition)
set(CMAKE_SYSTEM_VERSION 10.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 OpenCV 路径
set(OpenCV_DIR "C:/Program Files/opencv/build")
find_package(OpenCV REQUIRED dnn)

# 更新 ONNX Runtime 包路径
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-win-x64-1.19.2")
include_directories("${ONNXRUNTIME_DIR}/include")
link_directories("${ONNXRUNTIME_DIR}/lib")

if(MSVC)
    add_compile_options("/utf-8")
    set(CMAKE_GENERATOR_TOOLSET "v142")
    set(CMAKE_VS_PLATFORM_TOOLSET "v142")
endif()

# 添加源文件
set(SOURCES
    face_recognition.cpp
)

add_library(face_recognition SHARED ${SOURCES})

# 设置包含目录
target_include_directories(face_recognition PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

# 设置链接目录
target_link_directories(face_recognition PRIVATE
    ${OpenCV_LIB_DIR}
    ${ONNXRUNTIME_DIR}/lib
)

# 链接库（包括 onnxruntime）
target_link_libraries(face_recognition 
    ${OpenCV_LIBS} 
    onnxruntime
)

# 复制 onnxruntime.dll 到输出目录
add_custom_command(TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:face_recognition>"
)

# 定义模型文件的源路径和目标路径
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
set(YOLOV5_MODEL_SOURCE "${ASSETS_DIR}/yolov5l.onnx")
set(ARCFACE_MODEL_SOURCE "${ASSETS_DIR}/arcface_model.onnx")
set(YOLOV5_MODEL_DEST "$<TARGET_FILE_DIR:face_recognition>/yolov5l.onnx")
set(ARCFACE_MODEL_DEST "$<TARGET_FILE_DIR:face_recognition>/arcface_model.onnx")

# 检查源文件是否存在
if(NOT EXISTS ${YOLOV5_MODEL_SOURCE})
    message(FATAL_ERROR "Source YOLOV5 model file not found: ${YOLOV5_MODEL_SOURCE}")
endif()
if(NOT EXISTS ${ARCFACE_MODEL_SOURCE})
    message(FATAL_ERROR "Source ArcFace model file not found: ${ARCFACE_MODEL_SOURCE}")
endif()

# 复制生成的 DLL 文件到 assets 目录
add_custom_command(
    TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:face_recognition>"
        "${ASSETS_DIR}/face_recognition.dll"
    COMMENT "Copying generated face_recognition.dll to assets directory"
)

# 打印一些调试信息
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
message(STATUS "DLL_DEST: ${ASSETS_DIR}/face_recognition.dll")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")

# 添加编译定义
target_compile_definitions(face_recognition PRIVATE ORT_API_MANUAL_INIT)
