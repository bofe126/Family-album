# Face Recognition Library

基于 YOLOv5 和 ArcFace 的人脸检测和特征提取库。

## 依赖项

- OpenCV 4.10.0
- ONNX Runtime 1.19.2
- Visual Studio 2022 (Windows)
- CMake 3.10+

## 构建步骤

1. 安装依赖项：
   ```bash
   # 下载并安装 OpenCV
   # 下载并解压 ONNX Runtime
   ```

2. 设置环境变量：
   ```bash
   set OpenCV_DIR=C:\Program Files\opencv\build
   ```

3. 构建项目：
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

## 使用方法
