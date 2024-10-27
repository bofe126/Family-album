# 检查 OpenCV 安装路径
if(NOT OpenCV_DIR)
    set(OpenCV_DIR "C:/Program Files/opencv/build" CACHE PATH "OpenCV installation directory")
endif()

# 检查必要的文件是否存在
if(NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    message(FATAL_ERROR "OpenCV configuration file not found at ${OpenCV_DIR}/OpenCVConfig.cmake")
endif()

# 检查必要的头文件
if(NOT EXISTS "${OpenCV_DIR}/include/opencv2/opencv.hpp")
    message(FATAL_ERROR "OpenCV main header not found at ${OpenCV_DIR}/include/opencv2/opencv.hpp")
endif()

# 检查必要的库文件
if(NOT EXISTS "${OpenCV_DIR}/x64/vc16/lib/opencv_world410.lib")
    message(FATAL_ERROR "OpenCV library not found at ${OpenCV_DIR}/x64/vc16/lib/opencv_world410.lib")
endif()

# 检查必要的 DLL 文件
if(NOT EXISTS "${OpenCV_DIR}/x64/vc16/bin/opencv_world410.dll")
    message(FATAL_ERROR "OpenCV DLL not found at ${OpenCV_DIR}/x64/vc16/bin/opencv_world410.dll")
endif()

# 设置变量
set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
set(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
set(OpenCV_LIB_DIR "${OpenCV_DIR}/x64/vc16/lib")

# 输出调试信息
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_BIN_DIR: ${OpenCV_BIN_DIR}")
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
