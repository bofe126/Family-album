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

if(MSVC)
  add_compile_options("/utf-8")
  # 或者使用下面这行来忽略警告
  # add_compile_options("/wd4819")
endif()

# 添加源文件
set(SOURCES
    face_recognition.cpp
)

add_library(face_recognition SHARED ${SOURCES})
target_link_libraries(face_recognition ${OpenCV_LIBS})

# 设置包含目录
target_include_directories(face_recognition PRIVATE ${OpenCV_INCLUDE_DIRS})

# 定义模型文件的源路径和目标路径
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../assets")
set(PROTOTXT_SOURCE "${ASSETS_DIR}/deploy.prototxt")
set(CAFFEMODEL_SOURCE "${ASSETS_DIR}/res10_300x300_ssd_iter_140000.caffemodel")
set(ARCFACE_MODEL_SOURCE "${ASSETS_DIR}/arcface_model.onnx")
set(PROTOTXT_DEST "$<TARGET_FILE_DIR:face_recognition>/deploy.prototxt")
set(CAFFEMODEL_DEST "$<TARGET_FILE_DIR:face_recognition>/res10_300x300_ssd_iter_140000.caffemodel")
set(ARCFACE_MODEL_DEST "$<TARGET_FILE_DIR:face_recognition>/arcface_model.onnx")

# 检查源文件是否存在
if(NOT EXISTS ${PROTOTXT_SOURCE})
    message(FATAL_ERROR "Source prototxt file not found: ${PROTOTXT_SOURCE}")
endif()
if(NOT EXISTS ${CAFFEMODEL_SOURCE})
    message(FATAL_ERROR "Source caffemodel file not found: ${CAFFEMODEL_SOURCE}")
endif()
if(NOT EXISTS ${ARCFACE_MODEL_SOURCE})
    message(FATAL_ERROR "Source ArcFace model file not found: ${ARCFACE_MODEL_SOURCE}")
endif()

# 添加自定义命令来复制文件
add_custom_command(
    TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:face_recognition>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROTOTXT_SOURCE}" "${PROTOTXT_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CAFFEMODEL_SOURCE}" "${CAFFEMODEL_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ARCFACE_MODEL_SOURCE}" "${ARCFACE_MODEL_DEST}"
    COMMENT "Copying model files to output directory"
)

# 添加检查目标文件是否成功复制的命令
add_custom_command(
    TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Checking if files were copied successfully:"
    COMMAND ${CMAKE_COMMAND} -E echo "PROTOTXT_DEST: ${PROTOTXT_DEST}"
    COMMAND ${CMAKE_COMMAND} -E echo "CAFFEMODEL_DEST: ${CAFFEMODEL_DEST}"
    COMMAND ${CMAKE_COMMAND} -E echo "ARCFACE_MODEL_DEST: ${ARCFACE_MODEL_DEST}"
)

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
message(STATUS "PROTOTXT_SOURCE: ${PROTOTXT_SOURCE}")
message(STATUS "CAFFEMODEL_SOURCE: ${CAFFEMODEL_SOURCE}")
message(STATUS "ARCFACE_MODEL_SOURCE: ${ARCFACE_MODEL_SOURCE}")
message(STATUS "DLL_DEST: ${ASSETS_DIR}/face_recognition.dll")
