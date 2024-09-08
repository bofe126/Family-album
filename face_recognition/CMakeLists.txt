cmake_minimum_required(VERSION 3.10)
project(face_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Program Files/opencv/build")
find_package(OpenCV REQUIRED dnn)

if(MSVC)
  add_compile_options("/utf-8")
  # 或者使用下面这行来忽略警告
  # add_compile_options("/wd4819")
endif()

add_library(face_recognition SHARED face_recognition.cpp)
target_link_libraries(face_recognition ${OpenCV_LIBS})

# 在文件末尾添加这些行
add_custom_command(TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/deploy.prototxt"
        "${CMAKE_CURRENT_SOURCE_DIR}/res10_300x300_ssd_iter_140000.caffemodel"
        $<TARGET_FILE_DIR:face_recognition>)