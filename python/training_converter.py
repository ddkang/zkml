# A converter for training data
# Performs the conversion npy -> msgpack
# TODO: Ensure that training works with models that take in multiple input shapes
# 
# Shortcut: 
# `python3 python/training_converter.py --input_shapes 7,7,320 --input_idxes 1,0 --output training_data/inputs.msgpack --labels_output training_data/labels.msgpack`
#

import argparse
import ast
import numpy as np
import msgpack
import os

NUM_LOADS = 1
SF = 1 << 17

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_shapes', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)

  TRAINING_DIRECTORY = './testing/data/pre_last_conv/flowers/train'
  args = parser.parse_args()

  input_shapes = ast.literal_eval(args.input_shapes)

  loaded = 0
  tensors = []
  num_classes = os.listdir(TRAINING_DIRECTORY)

  first_file = "0.npy"
  for file_name in os.listdir(TRAINING_DIRECTORY):
    if loaded == NUM_LOADS:
      break

    label = int(first_file[:-4])
    data_array = np.load(TRAINING_DIRECTORY + '/' + first_file)

    input_shape = input_shapes

    for idx in range(data_array.shape[0]):
      print(SF)
      print((np.vstack(data_array) * SF).round().astype(np.int64))
      tensors.append({
        'idx': 0,
        'shape': input_shape,
        'data': list(map(lambda x: int(x), list((data_array[idx] * SF).round().astype(np.int64).flatten()))),
      })
      # represent the label as a one hot encoding
      one_hot = np.zeros(102)
      one_hot[label] = SF
      print("IMPORTANT LABEL", label)
      print("IMPORTANT LABEL", data_array[idx].flatten()[:500])
      # print(one_hot.shape())
      tensors.append({
        'idx': 11,
        'shape': (1, 102),
        'data': list(map(lambda x: int(x), one_hot)),
      })
      loaded += 1

      if loaded == NUM_LOADS:
        break

  packed_inputs = msgpack.packb(tensors, use_bin_type=True)

  # print(tensors)
  with open(args.output, 'wb') as f:
    f.write(packed_inputs)

if __name__ == '__main__':
  main()
