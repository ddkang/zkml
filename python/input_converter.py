import argparse
import ast
import numpy as np
import msgpack

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs', type=str, required=True)
  parser.add_argument('--input_shapes', type=str, required=True)
  parser.add_argument('--input_idxes', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()

  inputs = args.inputs.split(',')
  input_shapes = ast.literal_eval(args.input_shapes)
  input_idxes = ast.literal_eval(args.input_idxes)

  assert len(inputs) == len(input_shapes)
  assert len(inputs) == len(input_idxes)

  tensors = []
  for inp, shape, idx in zip(inputs, input_shapes, input_idxes):
    tensor = np.fromfile(inp, dtype=np.int64).reshape(shape)
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