# 人脸识别应用

这个Flutter应用使用YOLOV5-face模型进行人脸检测，并使用ArcFace模型进行人脸特征提取。支持Windows桌面平台和Chrome Web平台。

## 主要功能

- 人脸检测和识别
- 照片管理和浏览
- 人脸搜索
- 相似度阈值调整

## 依赖

- Flutter SDK
- Visual Studio（用于Windows C++开发）
- CMake
- OpenCV
- ONNX Runtime
- YOLOV5-face model
- ArcFace model
- Windows平台支持
- Chrome Web平台支持

## 安装和构建步骤

1. 克隆仓库：
   ```bash
   git clone [您的项目URL]
   cd [项目目录]
   ```

2. 安装Flutter依赖：
   ```bash
   flutter pub get
   ```

3. 编译C++ DLL（仅Windows平台）：
   - 打开Visual Studio Developer Command Prompt
   - 导航到项目的C++源代码目录（假设在 `windows/cpp_source`）
   ```bash
   cd windows/cpp_source
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ```
   - 这将生成 `face_recognition.dll`

   注意：在Windows平台上，构建类型（Debug/Release）是在 `cmake --build` 命令中通过 `--config` 参数指定的，而不是在初始的 `cmake` 命令中。

4. 复制资源文件（仅Windows平台）：
   - 将 `face_recognition.dll`、`yolov5-face.onnx` 和 `arcface_model.onnx` 复制到 `assets` 文件夹

5. 构建Windows应用：
   ```bash
   flutter build windows
   ```

6. 构建Web应用（Chrome支持）：
   ```bash
   flutter build web
   ```

7. 运行应用：
   - Windows：
     ```bash
     flutter run -d windows
     ```
   - Chrome：
     ```bash
     flutter run -d chrome
     ```

## 使用

1. 启动应用
2. 选择或拍摄照片进行人脸检测
3. 使用侧边栏浏览检测到的人脸
4. 调整相似度阈值以优化识别结果

## 注意事项

- 本应用支持Windows桌面平台和Chrome Web平台
- 核心算法使用C++实现，通过FFI与Flutter交互（仅Windows平台）
- Web版本可能不支持某些依赖于原生代码的功能（如C++ DLL交互）
- 请确保所有依赖模型和DLL文件都正确放置在assets文件夹中（仅Windows平台）
- 确保 `windows/runner/main.cpp` 正确加载 DLL 和初始化 Flutter 引擎（仅Windows平台）
- 在 `lib/main.dart` 中使用 `dart:ffi` 来调用 C++ 函数（仅Windows平台）

## 调试

- 使用Visual Studio Code的Flutter插件进行Flutter部分的调试
- 对于C++部分，可以使用Visual Studio进行调试（仅Windows平台）
- 使用Chrome开发者工具调试Web版本

## 贡献

欢迎提交问题和拉取请求。

## 许可

[在此添加许可信息]