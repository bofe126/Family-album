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

:: 检查是否需要重新配置
set NEED_CONFIGURE=0
if not exist "build\CMakeCache.txt" (
    set NEED_CONFIGURE=1
    echo Build directory does not exist or is empty. Will configure...
) else (
    :: 检查关键文件的修改时间
    for %%F in (CMakeLists.txt cmake\*.cmake include\*.h src\*.cpp) do (
        if %%~tF gtr build\CMakeCache.txt (
            set NEED_CONFIGURE=1
            echo File %%F has been modified. Will reconfigure...
            goto :configure
        )
    )
)

:configure
:: 只在需要时重新配置
if %NEED_CONFIGURE%==1 (
    echo Configuring project...
    if not exist build mkdir build
    cd build
    cmake -G "Visual Studio 17 2022" -A x64 ^
        -DOpenCV_DIR="%OpenCV_DIR%" ^
        -DCMAKE_BUILD_TYPE=Debug ^
        ..
    if errorlevel 1 (
        echo Error: CMake configuration failed
        cd ..
        exit /b 1
    )
    cd ..
) else (
    echo Using existing CMake configuration...
)

:: 构建项目
echo Building project in Debug mode...
cmake --build build --config Debug --parallel %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo Error: Build failed
    exit /b 1
)

:: 确保 assets 目录存在
if not exist "..\assets" mkdir "..\assets"

:: 只复制 DLL 和 PDB 到 assets 目录
echo Copying files to assets directory...
copy /Y "build\bin\Debug\face_recognition.dll" "..\assets\face_recognition.dll"
copy /Y "build\bin\Debug\face_recognition.pdb" "..\assets\face_recognition.pdb"

echo Build completed successfully in Debug mode.
echo Files have been copied to assets directory.
pause
