import onnx

model = onnx.load("./yolov5lite-g.onnx")
print(f"ONNX IR version: {model.ir_version}")
print(f"Opset version: {model.opset_import[0].version}")

# 打印模型的输入
print("Model Inputs:")
for input in model.graph.input:
        print(f"  Name: {input.name}")
        print(f"  Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

# 打印模型的输出
        print("Model Outputs:")
        for output in model.graph.output:
            print(f"  Name: {output.name}")
            print(f"  Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
