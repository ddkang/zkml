import argparse
import ast
import numpy as np
import msgpack

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_config', type=str, required=True)
  parser.add_argument('--inputs', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()

  inputs = args.inputs.split(',')
  with open(args.model_config, 'rb') as f:
    model_config = msgpack.unpackb(f.read())
  input_idxes = model_config['inp_idxes']
  scale_factor = model_config['global_sf']

  # Get the input shapes from the layers
  input_shapes = [[0] for _ in input_idxes]
  for layer in model_config['layers']:
    for layer_inp_idx, layer_shape in zip(layer['inp_idxes'], layer['inp_shapes']):
      for index, inp_idx in enumerate(input_idxes):
        if layer_inp_idx == inp_idx:
          input_shapes[index] = layer_shape

  tensors = []
  for inp, shape, idx in zip(inputs, input_shapes, input_idxes):
    tensor = np.load(inp).reshape(shape)
    tensor = (tensor * scale_factor).round().astype(np.int64)
    tensors.append({
      'idx': idx,
      'shape': shape,
      'data': tensor.flatten().tolist(),
    })

  packed = msgpack.packb(tensors, use_bin_type=True)

  with open(args.output, 'wb') as f:
    f.write(packed)


if __name__ == '__main__':
  main()