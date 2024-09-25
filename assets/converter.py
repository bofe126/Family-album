import onnx
from onnx import version_converter

model = onnx.load("yolov5lite-g.onnx")
converted_model = version_converter.convert_version(model, 7)  # 转换到 opset 10
onnx.save(converted_model, "yolov5lite-g_converted.onnx")
