# 模型信息

本目录包含了人脸识别应用所需的模型文件。

## YOLO 模型

- 文件名: yolov5s-face.onnx
- 输入:
  - 名称: input
  - 形状: [1, 3, 640, 640]
  - 类型: tensor(float)
- 输出:
  - 名称: output
  - 形状: [1, 25200, 16]
  - 类型: tensor(float)

## ArcFace 模型

- 文件名: arcface_model.onnx
- 输入/输出信息: [待补充]

## 注意事项

- YOLO 模型的输入图像应该被调整为 640x640 像素。
- 输出张量包含 25200 个检测结果，每个结果由 16 个值组成。
- 确保这些模型文件与 face_recognition.dll 放在同一目录下。
- 在使用模型之前，请确保您有使用这些模型的适当许可。

## 更新日志

- [日期]: 添加 yolov5s-face.onnx 模型
- [日期]: 添加 arcface_model.onnx 模型

请根据实际情况更新此文件，特别是 ArcFace 模型的详细信息和更新日志。
