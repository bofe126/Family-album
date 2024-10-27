@echo off
setlocal

:: 设置环境变量
set "OpenCV_DIR=C:\Program Files\opencv\build"
set "PATH=%PATH%;%OpenCV_DIR%\x64\vc16\bin"

:: 检查 OpenCV 目录是否存在
if not exist "%OpenCV_DIR%" (
    echo Error: OpenCV directory not found at %OpenCV_DIR%
    exit /b 1
)

:: 检查必要的 OpenCV 文件
if not exist "%OpenCV_DIR%\include\opencv2\opencv.hpp" (
    echo Error: OpenCV headers not found
    exit /b 1
)

:: 删除旧的构建目录
if exist build (
    echo Removing old build directory...
    rd /s /q build
)

:: 创建并进入构建目录
echo Creating new build directory...
mkdir build
cd build

:: 配置项目
echo Configuring project...
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DOpenCV_DIR="%OpenCV_DIR%" ^
    ..

if errorlevel 1 (
    echo Error: CMake configuration failed
    cd ..
    exit /b 1
)

:: 构建项目
echo Building project...
cmake --build . --config Release

if errorlevel 1 (
    echo Error: Build failed
    cd ..
    exit /b 1
)

:: 返回原目录
cd ..

echo Build completed successfully.
pause
