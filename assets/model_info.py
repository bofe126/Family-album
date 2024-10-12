import onnxruntime as ort

model_path = "yolov5s-face.onnx"
session = ort.InferenceSession(model_path)

# 获取输入信息
inputs = session.get_inputs()
for input in inputs:
    print(f"Input Name: {input.name}")
    print(f"Input Shape: {input.shape}")
    print(f"Input Type: {input.type}")

# 获取输出信息
outputs = session.get_outputs()
for output in outputs:
    print(f"Output Name: {output.name}")
    print(f"Output Shape: {output.shape}")
    print(f"Output Type: {output.type}")
