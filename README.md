# 人脸识别应用

这个Flutter应用使用YOLOV5-face模型进行人脸检测，并使用ArcFace模型进行人脸特征提取。

## 依赖

- Flutter
- OpenCV
- ONNX Runtime
- YOLOV5-face model
- ArcFace model

## 安装

1. 克隆仓库
2. 运行 `flutter pub get` 安装依赖
3. 确保 `assets` 文件夹中包含以下文件：
   - `yolov5-face.onnx`
   - `arcface_model.onnx`
   - `face_recognition.dll`

## 使用

...