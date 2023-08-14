# Oggn
import onnx
import numpy as np
import msgpack
from onnx import numpy_helper

scale_factor = 512

# Loading ONNX model

onnx_model = onnx.load("mnist-8.onnx")
model_graph = onnx_model.graph
model_input = model_graph.input
model_nodes = model_graph.node
model_init = onnx_model.graph.initializer

# Helper Functions


def get_shape(container, node_id):
  dim = container.__getitem__(node_id).type.tensor_type.shape.dim
  dim_list = list()
  for i in range(len(dim)):
    dim_list.append(dim.pop(0).dim_value)
  return dim_list


  for i in range(len(dim)):
    dim_list.append(dim.pop(i).dim_value)
  return dim_list


def get_output_dim(node_id, graph):
  model_output = graph.output
  model_valueinfo = graph.value_info
  if node_id == len(graph.node) - 1:
    output_dim = get_shape(model_output, 0)
  else:
    output_dim = get_shape(model_valueinfo, node_id)
  return output_dim


def get_input_dim(layers):
  return layers[-1]['output_shapes']


# Converting Layers

layers = list()
node_id = 0
for node in model_nodes:

  if node.op_type == "Conv":
    layer_type = "Conv2D"
    output_id = node_id
    if node_id == 0:
      input_dim = get_shape(model_input, 0)
    else:
      input_dim = get_input_dim(layers)
    output_dim = get_output_dim(node_id, model_graph)
    node_attr = node.attribute
    kernel = [_ for _ in node_attr.pop(0).ints]
    stride = [_ for _ in node_attr.pop(0).ints]
    # padding = str(node_attr.pop(0).s) # TODO
    params = [kernel, stride]
  elif node.op_type == "MaxPool":
    layer_type = "MaxPool2D"
    output_id = node_id
    if node_id == 0:
      input_dim = get_shape(model_input, 0)
    else:
      input_dim = get_input_dim(layers)
    output_dim = get_output_dim(node_id, model_graph)
    node_attr = node.attribute
    kernel = [_ for _ in node_attr.pop(0).ints]
    stride = [_ for _ in node_attr.pop(0).ints]
    params = [kernel, stride]

  elif node.op_type == "Relu":
    layer_type = "ReLU"
    output_id = node_id
    if node_id == 0:
      input_dim = get_shape(model_input, 0)
    else:
      input_dim = get_input_dim(layers)
    output_dim = get_output_dim(node_id, model_graph)
    params = None

  elif node.op_type == "Reshape":
    layer_type = "Reshape"
    output_id = node_id
    if node_id == 0:
      input_dim = get_shape(model_input, 0)
    else:
      input_dim = get_input_dim(layers)
    output_dim = get_output_dim(node_id, model_graph)
    params = None

  elif node.op_type == "Gemm":
    layer_type = "Fully Connected Layer"
    output_id = node_id
    if node_id == 0:
      input_dim = get_shape(model_input, 0)
    else:
      input_dim = get_input_dim(layers)
    output_dim = get_output_dim(node_id, model_graph)
    params = None
  else:
    node_id += 1
    continue
  layer = {
    "layer_type": layer_type,
    "node_id": node_id,
    "input_shapes": input_dim,
    "output_shapes": output_dim,
    "params": params
  }
  layers.append(layer)
  node_id += 1
# print(layers)

# Converting W&B

tensors = list()
for init in model_init:
  shape = [dim for dim in init.dims]
  raw_data = numpy_helper.to_array(init).ravel().tolist()  ## Orientation of ravel ?????
  data = []
  for i in raw_data:
    if isinstance(i, float):
      buf = int(np.round(i * scale_factor))
      data.append(buf)
  tensor = {"shape": shape, "data": data}
  tensors.append(tensor)
# print(tensors)

# Converting to msgpack

final_dict = {
  "scaling_factor": scale_factor,
  "layers": layers,
  "tensors": tensors
}

with open("first_transformed_onnx.msgpack", "wb") as mfile:
  mfile.write(msgpack.packb(final_dict))
