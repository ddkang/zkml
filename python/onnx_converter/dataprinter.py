import onnx

onnx_model = onnx.load("mnist-8.onnx")

print(onnx_model.graph.value_info)

